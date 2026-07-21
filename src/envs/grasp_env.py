"""
GraspEnv — Standard Gym-style RL environment for robotic grasping.

Follows official Genesis manipulation/grasp_env.py pattern:
- Scene + robot + objects created internally
- GPU tensors throughout (no CPU/numpy in hot path)
- Standard Gym interface: reset() → TensorDict, step() → (obs, reward, done, info)
- Direct manipulation methods for scripted control (pick/place/execute_action)

Usage (RL training):
    env = GraspEnv(num_envs=64)
    obs = env.reset()
    obs, reward, done, info = env.step(action)  # action: [B, 6] delta EE

Usage (scripted demo):
    env = GraspEnv()
    env.execute_action({"action": "pick", "object": "red_cube"})
    env.execute_action({"action": "place", "target": "blue_cube"}, target_pos=[0.4,0.2,0.02])
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import genesis as gs


# ── Manipulator (robot wrapper) ────────────────────────────────

class Manipulator:
    """Franka Panda robot wrapper. All operations on GPU tensors."""

    def __init__(self, scene: gs.Scene, device: str = "cpu"):
        self._scene = scene
        self._device = device

        self._robot = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        self._n_arm = 7
        self._gripper_dim = 2
        self._arm_dof_idx = torch.arange(self._n_arm, device=device)
        self._fingers_dof = torch.arange(
            self._n_arm, self._n_arm + self._gripper_dim, device=device)
        self._gripper_open = 0.04
        self._gripper_close = 0.00
        self._ee_link = self._robot.get_link("hand")

        self._init_qpos = torch.tensor(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            dtype=torch.float32, device=device,
        )

    @property
    def entity(self):
        return self._robot

    @property
    def ee_link(self):
        return self._ee_link

    @property
    def n_dofs(self) -> int:
        return self._robot.n_dofs

    @property
    def ee_pose(self) -> torch.Tensor:
        pos = self._ee_link.get_pos()
        quat = self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    def set_pd_gains(self):
        self._robot.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self._robot.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self._robot.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def reset(self, envs_idx=None):
        self._robot.set_qpos(
            self._init_qpos, envs_idx=envs_idx,
            zero_velocity=True, skip_forward=True)

    def apply_action(self, action: torch.Tensor, open_gripper: bool = True):
        """Apply 6D delta-EE action via Genesis IK. action: ([B,] 6)"""
        delta_pos = action[..., :3]
        delta_rpy = action[..., 3:6]

        target_pos = delta_pos + self._ee_link.get_pos()
        from genesis.utils.geom import xyz_to_quat, transform_quat_by_quat
        quat_rel = xyz_to_quat(delta_rpy, rpy=True, degrees=False)
        target_quat = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())

        q_pos = self._robot.inverse_kinematics(
            link=self._ee_link, pos=target_pos, quat=target_quat,
            dofs_idx_local=self._arm_dof_idx,
        )
        q_pos[..., self._fingers_dof] = (
            self._gripper_open if open_gripper else self._gripper_close)
        self._robot.control_dofs_position(q_pos)

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        """Move EE to absolute pose. goal_pose: ([B,] 7) pos+quat."""
        q_pos = self._robot.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[..., :3], quat=goal_pose[..., 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        q_pos[..., self._fingers_dof] = (
            self._gripper_open if open_gripper else self._gripper_close)
        self._robot.control_dofs_position(q_pos)

    def solve_ik(self, pos, quat=None):
        """Solve IK for target position. Returns full qpos tensor."""
        if quat is None:
            quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32,
                                device=self._device)
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, dtype=torch.float32, device=self._device)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float32, device=self._device)
        return self._robot.inverse_kinematics(
            link=self._ee_link, pos=pos, quat=quat)


# ── GraspEnv ───────────────────────────────────────────────────

class GraspEnv:
    """Gym-style grasping environment following official Genesis pattern.

    Creates scene, robot, objects, and cameras internally.
    Supports both RL training (batched) and scripted control (single env).
    """

    def __init__(
        self,
        num_envs: int = 0,
        device: str = None,
        ctrl_dt: float = 0.01,
        episode_length_s: float = 10.0,
        action_scale: float = 0.05,
        image_size: Tuple[int, int] = (64, 64),
    ):
        self.num_envs = num_envs
        self.num_actions = 6  # 6D delta-EE
        self.device = device or str(gs.device)
        self.ctrl_dt = ctrl_dt
        self.action_scale = action_scale
        self.max_episode_length = math.ceil(episode_length_s / ctrl_dt)
        self.image_size = image_size
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.cfg = {
            "num_envs": num_envs,
            "num_actions": 6,
            "ctrl_dt": ctrl_dt,
            "episode_length_s": episode_length_s,
            "action_scale": action_scale,
        }

        self._build_scene()
        self._build_objects()
        self._setup_cameras()

        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))
        self.robot.set_pd_gains()

        self._init_buffers()
        self.reset()

    # ── Scene construction ──────────────────────────────────────

    def _build_scene(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = Manipulator(self.scene, device=self.device)
        self.rigid_solver = self.scene.sim.rigid_solver

    def _build_objects(self):
        """Build scene objects. Supports cubes, cylinders, and spheres."""
        s = 0.04  # standard object size
        objects = [
            # (name, morph, color, position)
            ("red_cube", gs.morphs.Box(size=(s, s, s), pos=(0.65, 0.0, s/2)),
             (1, 0, 0), (0.65, 0.0, s/2)),
            ("blue_cube", gs.morphs.Box(size=(s, s, s), pos=(0.4, 0.2, s/2)),
             (0, 1, 0), (0.4, 0.2, s/2)),
            ("green_cube", gs.morphs.Box(size=(s, s, s), pos=(0.7, -0.1, s/2)),
             (0, 0, 1), (0.7, -0.1, s/2)),
            ("yellow_cylinder", gs.morphs.Cylinder(radius=s/2, height=s, pos=(0.55, 0.15, s/2)),
             (1, 1, 0), (0.55, 0.15, s/2)),
            ("purple_sphere", gs.morphs.Sphere(radius=s/2, pos=(0.5, -0.15, s/2)),
             (0.5, 0, 0.5), (0.5, -0.15, s/2)),
            # Target zone indicator (thin blue square on ground)
            ("target_zone", gs.morphs.Box(size=(0.06, 0.06, 0.002), pos=(0.46, 0.20, 0.001)),
             (0.3, 0.5, 1.0), (0.46, 0.20, 0.001)),
        ]
        self.entities: Dict[str, object] = {}
        self._original_positions: Dict[str, torch.Tensor] = {}
        for name, morph, color, pos in objects:
            ent = self.scene.add_entity(morph, surface=gs.surfaces.Plastic(color=color))
            self.entities[name] = ent
            self._original_positions[name] = torch.tensor(
                pos, dtype=torch.float32, device=self.device)

    def _setup_cameras(self):
        from genesis.options.sensors import RasterizerCameraOptions
        try:
            self.left_cam = self.scene.add_sensor(RasterizerCameraOptions(
                res=self.image_size,
                pos=(1.25, 0.3, 0.3), lookat=(0.0, 0.0, 0.0), fov=60,
            ))
            self.right_cam = self.scene.add_sensor(RasterizerCameraOptions(
                res=self.image_size,
                pos=(1.25, -0.3, 0.3), lookat=(0.0, 0.0, 0.0), fov=60,
            ))
        except Exception:
            self.left_cam = None
            self.right_cam = None

        self.vis_cam = self.scene.add_camera(
            res=(1280, 720), pos=(1.5, -2.0, 1.6),
            lookat=(0.5, 0.0, 0.0), fov=45, GUI=False,
        )

    def _init_buffers(self):
        B = max(self.num_envs, 1)
        self.episode_length_buf = torch.zeros(
            B, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(
            B, device=self.device, dtype=torch.bool)
        self.extras: Dict = {}

        # Reward scales (dt-weighted composite reward)
        self.reward_scales = {
            "approach": 0.3 * self.ctrl_dt,  # Keypoint proximity
            "grasp": 0.3 * self.ctrl_dt,     # Object lifted off ground
            "lift": 0.2 * self.ctrl_dt,      # Continuous lift height
            "place": 0.2 * self.ctrl_dt,     # Distance to target
        }
        self.episode_sums = {
            k: torch.zeros(B, device=self.device, dtype=torch.float32)
            for k in self.reward_scales
        }
        self.reward_functions = {
            k: getattr(self, "_reward_" + k) for k in self.reward_scales
        }

        # Keypoint offsets (7 points: origin + 6 axis-aligned)
        unit = 0.5
        offsets = torch.tensor([
            [0, 0, 0], [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
        ], device=self.device, dtype=torch.float32) * unit
        self.keypoints_offset = offsets.unsqueeze(0).expand(
            max(self.num_envs, 1), -1, -1).clone()

        # Target object (first entity by default)
        self._target_obj = list(self.entities.keys())[0]

    # ── Gym interface (official grasp_env.py pattern) ─────────

    def reset(self, envs_idx=None):
        """Reset robot + objects. Returns initial observation."""
        self.robot.reset(envs_idx)
        for name, orig_pos in self._original_positions.items():
            self.entities[name].set_pos(
                orig_pos, envs_idx=envs_idx, skip_forward=True)
        if envs_idx is not None:
            self.entities[self._target_obj].set_quat(
                torch.tensor([0, 1, 0, 0], dtype=torch.float32,
                             device=self.device).expand(
                    (1 if envs_idx.dim() == 1 else envs_idx.sum()), -1),
                envs_idx=envs_idx, skip_forward=False)
        self.scene.step(20)

        if envs_idx is None:
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            for v in self.episode_sums.values():
                v.zero_()
        else:
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, False)
            for v in self.episode_sums.values():
                v.masked_fill_(envs_idx, 0.0)

        return self.get_observations()

    def step(self, actions: torch.Tensor):
        """Apply delta-EE actions. actions: ([B,] 6) or ([B,] 7)."""
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions[..., :6], open_gripper=True)
        self.scene.step()

        self.episode_length_buf += 1
        self.reset_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.scene.sim.rigid_solver.get_error_envs_mask()

        self.extras["time_outs"] = (
            self.episode_length_buf >= self.max_episode_length
        ).to(dtype=torch.float32)

        # Compute reward
        reward = torch.zeros(max(self.num_envs, 1), device=self.device,
                             dtype=torch.float32)
        for name, rfunc in self.reward_functions.items():
            rew = rfunc() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        self.extras["episode"] = {}
        for k, v in self.episode_sums.items():
            self.extras["episode"][f"rew_{k}"] = v.mean()

        if self.reset_buf.any():
            self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, self.reset_buf, self.extras

    def _reset_idx(self, envs_idx):
        self.robot.reset(envs_idx)
        for name, orig_pos in self._original_positions.items():
            self.entities[name].set_pos(
                orig_pos, envs_idx=envs_idx, skip_forward=True)
        self.episode_length_buf.masked_fill_(envs_idx, 0)
        self.reset_buf.masked_fill_(envs_idx, False)
        n = envs_idx.sum() if envs_idx is not None else self.num_envs
        for k, v in self.episode_sums.items():
            if envs_idx is None:
                self.extras.setdefault("episode", {})[f"rew_{k}"] = v.mean()
            else:
                self.extras.setdefault("episode", {})[f"rew_{k}"] = (
                    v[envs_idx].sum() / max(n, 1))
            v.masked_fill_(envs_idx, 0.0)

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale

    # ── Observations (14-dim, official pattern) ────────────────

    def get_observations(self) -> "TensorDict":
        """14-dim obs: (finger_pos-obj_pos)[3], finger_quat[4], obj_pos[3], obj_quat[4]."""
        from tensordict import TensorDict
        finger_pos = self.robot.ee_link.get_pos()
        finger_quat = self.robot.ee_link.get_quat()
        obj = self.entities[self._target_obj]
        obj_pos, obj_quat = obj.get_pos(), obj.get_quat()
        obs = torch.cat([
            finger_pos - obj_pos,  # 3
            finger_quat,           # 4
            obj_pos,               # 3
            obj_quat,              # 4
        ], dim=-1)  # total 14
        # Ensure batch dimension for single-env mode
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return TensorDict({"policy": obs}, batch_size=[obs.shape[0]])

    # ── Reward (composite: approach + grasp + lift + place) ────

    def _ensure_batch(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has batch dimension."""
        if t.dim() == 1:
            return t.unsqueeze(0)
        return t

    def _reward_approach(self) -> torch.Tensor:
        """Approach reward: exp(-keypoint_distance)."""
        from genesis.utils.geom import transform_by_trans_quat

        B = max(self.num_envs, 1)
        finger_tip_offset = torch.tensor(
            [0.0, 0.0, -0.06], device=self.device, dtype=torch.float32,
        ).expand(B, -1)

        finger_pos = self._ensure_batch(self.robot.ee_link.get_pos())
        finger_quat = self._ensure_batch(self.robot.ee_link.get_quat())
        obj = self.entities[self._target_obj]
        obj_pos = self._ensure_batch(obj.get_pos())
        obj_quat = self._ensure_batch(obj.get_quat())

        finger_pos = finger_pos + finger_tip_offset

        finger_kp = transform_by_trans_quat(
            self.keypoints_offset,
            finger_pos.unsqueeze(1),
            finger_quat.unsqueeze(1),
        )
        obj_kp = transform_by_trans_quat(
            self.keypoints_offset,
            obj_pos.unsqueeze(1),
            obj_quat.unsqueeze(1),
        )
        dist = torch.norm(finger_kp - obj_kp, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    def _reward_grasp(self) -> torch.Tensor:
        """Grasp reward: +1 when object is lifted above ground."""
        obj_pos = self._ensure_batch(self.entities[self._target_obj].get_pos())
        # Object lifted if z > initial_z + 0.03 (significant lift)
        initial_z = self._original_positions[self._target_obj][2]
        lifted = (obj_pos[..., 2] > initial_z + 0.03).float()
        return lifted

    def _reward_lift(self) -> torch.Tensor:
        """Lift reward: continuous bonus for lifting higher."""
        obj_pos = self._ensure_batch(self.entities[self._target_obj].get_pos())
        initial_z = self._original_positions[self._target_obj][2]
        # Linear reward: 0 at initial, 1.0 at z=0.30
        lift_height = torch.clamp(obj_pos[..., 2] - initial_z, 0, 0.30)
        return lift_height / 0.30

    def _reward_place(self) -> torch.Tensor:
        """Place reward: exp(-distance_to_target) when object is above ground."""
        obj_pos = self._ensure_batch(self.entities[self._target_obj].get_pos())
        target = torch.tensor([0.4, 0.2], dtype=torch.float32, device=self.device)
        dist_2d = torch.norm(obj_pos[..., :2] - target, dim=-1)
        # Only reward if object is above ground (grasped/lifted)
        above_ground = obj_pos[..., 2] > 0.04
        return torch.where(above_ground, torch.exp(-dist_2d * 5.0), torch.zeros_like(dist_2d))

    # ── Direct manipulation (for scripted control / demo) ──────

    def _solve_ik(self, x, y, z):
        """Solve IK using Genesis built-in solver (official pattern).
        
        Genesis's inverse_kinematics already handles IK solving internally.
        We just need to call it correctly and use the result directly.
        """
        franka = self.robot.entity
        ee = franka.get_link("hand")
        target = np.array([x, y, z], dtype=np.float64)
        
        # Use Genesis built-in IK (official pattern from tutorial)
        qpos = franka.inverse_kinematics(
            link=ee,
            pos=target.reshape(1, 3),
            quat=np.array([[0, 1, 0, 0]], dtype=np.float64),  # Gripper facing down
        )
        if qpos.dim() == 2:
            qpos = qpos[0]
        
        return qpos

    def _pd_hold(self, qpos_arm, steps):
        """Hold target qpos for `steps` (Genesis official pattern).
        
        Genesis official: control_dofs_position called ONCE + scene.step() N times
        """
        franka = self.robot.entity
        motors_dof = np.arange(7)
        franka.control_dofs_position(qpos_arm, motors_dof)
        for _ in range(steps):
            self.scene.step()

    def _render_frame(self):
        """Render a frame if recording is active."""
        if hasattr(self, 'vis_cam') and self.vis_cam is not None:
            try:
                self.vis_cam.render()
            except Exception:
                pass

    def suction_pick(self, obj_name: str, camera=None) -> bool:
        """Pick object using suction (weld constraint). Returns lifted.
        
        Follows Genesis official pattern from tutorial:
        1. IK to above object (0.25m)
        2. plan_path with collision avoidance (OMPL)
        3. PD descent to grasp height (0.130m per official)
        4. Weld constraint (suction)
        5. PD lift to safe height
        """
        franka = self.robot.entity
        ee = franka.get_link("hand")
        motors_dof = np.arange(7)
        fingers_dof = np.arange(7, 9)
        obj = self.entities[obj_name]
        obj_pos = np.asarray(obj.get_pos().cpu().numpy()).flatten()[:3]
        self._current_object = obj_name

        # ═══ Step 1: IK to above object ═══
        # Genesis official: use 0.25m above object
        above_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.25]
        qpos = self._solve_ik(above_pos[0], above_pos[1], above_pos[2])
        qpos[-2:] = 0.04  # Open gripper (Genesis official)

        # ═══ Step 2: plan_path with collision avoidance ═══
        # Genesis official: plan_path uses OMPL for collision-free path
        path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)  # Genesis official: 200 waypoints
        for waypoint in path:
            franka.control_dofs_position(waypoint)
            franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
            self.scene.step()
            self._render_frame()  # Record frame for video

        # Let PD controller converge to final waypoint (Genesis official)
        for _ in range(100):
            self.scene.step()
            self._render_frame()  # Record frame for video

        # ═══ Step 3: PD descent to grasp height ═══
        # Genesis official: use 0.130m (not obj_pos[2] + 0.05)
        # This is the precise grasp height for a 4cm cube on ground
        grasp_height = 0.130  # Genesis official value
        qpos = self._solve_ik(obj_pos[0], obj_pos[1], grasp_height)
        franka.control_dofs_position(qpos[:-2], motors_dof)

        # Wait for PD controller to converge (Genesis official: 100 steps)
        for _ in range(100):
            self.scene.step()
            self._render_frame()  # Record frame for video

        # ═══ Step 4: Weld constraint (suction) ═══
        self.rigid_solver.add_weld_constraint(
            obj.get_link("box_baselink").idx, ee.idx)
        self._grasped = True

        # ═══ Step 5: PD lift ═══
        # Genesis official: lift to 0.28m
        lift_height = 0.28  # Genesis official value
        qpos = self._solve_ik(obj_pos[0], obj_pos[1], lift_height)
        franka.control_dofs_position(qpos[:-2], motors_dof)

        # Wait for PD controller to converge (Genesis official: 100 steps)
        for _ in range(100):
            self.scene.step()
            self._render_frame()  # Record frame for video

        # Check if object was lifted
        final_z = float(np.asarray(obj.get_pos().cpu().numpy()).flatten()[2])
        return final_z > obj_pos[2] + 0.05  # Genesis official: 5cm lift threshold

    def suction_place(self, obj_name: str, target_pos, camera=None) -> float:
        """Place object at target. Returns XY error in meters.
        
        Follows Genesis official pattern from tutorial:
        1. IK to above target (0.18m)
        2. PD descent to place height
        3. Release weld constraint
        4. Wait for physics to settle
        """
        franka = self.robot.entity
        ee = franka.get_link("hand")
        motors_dof = np.arange(7)
        obj = self.entities[obj_name]

        if not isinstance(target_pos, (list, np.ndarray)):
            target_pos = target_pos.cpu().numpy().tolist()
        tp = np.array(target_pos, dtype=np.float64)
        self._current_target = tp.tolist()
        cube_link_idx = obj.get_link("box_baselink").idx
        hand_link_idx = ee.idx

        # ═══ Step 1: IK to above target ═══
        # Genesis official: use 0.18m above target
        above_height = 0.18  # Genesis official value
        qpos = self._solve_ik(tp[0], tp[1], tp[2] + above_height)
        franka.control_dofs_position(qpos[:-2], motors_dof)

        # Wait for PD controller to converge (Genesis official: 100 steps)
        for _ in range(100):
            self.scene.step()
            self._render_frame()  # Record frame for video

        # ═══ Step 2: Release weld constraint ═══
        # Genesis official: delete_weld_constraint → arm holds position → object settles
        self.rigid_solver.delete_weld_constraint(cube_link_idx, hand_link_idx)
        self._grasped = False

        # ═══ Step 3: Wait for physics to settle ═══
        # Genesis official: 400 steps for settling
        for i in range(400):
            self.scene.step()
            self._render_frame()  # Record frame for video

        # ═══ Step 4: Verify final position ═══
        final = np.asarray(obj.get_pos().cpu().numpy()).flatten()[:3]
        err = float(np.sqrt((final[0] - tp[0]) ** 2 + (final[1] - tp[1]) ** 2))
        return err

    def execute_action(self, action: Dict,
                       target_pos=None) -> Dict:
        """Dispatch action dict from ActionScheduler.

        Supports: pick, place, move_above, wait/settle.
        """
        act = action.get("action", "").lower()
        obj = action.get("object", "")
        tgt = action.get("target", "")

        if act == "pick":
            if obj in self.entities:
                lifted = self.suction_pick(obj)
                return {"ok": lifted, "result": lifted}
            return {"ok": False, "result": False,
                    "reason": f"unknown object {obj}"}

        if act == "place":
            if not obj:
                obj = self._current_object or list(self.entities.keys())[0]
            if obj not in self.entities:
                return {"ok": False, "result": 0.99,
                        "reason": f"unknown object {obj}"}
            if target_pos is not None:
                tp = target_pos
            elif tgt in self.entities:
                tp = self.entities[tgt].get_pos().tolist()
            else:
                tp = [0.55, 0.0, 0.02]
            err = self.suction_place(obj, tp)
            return {"ok": err < 0.10, "result": err}

        if act == "move_above":
            name = obj or tgt
            if name in self.entities:
                franka = self.robot.entity
                pos = np.asarray(self.entities[name].get_pos().cpu().numpy()).flatten()[:3]
                q = franka.inverse_kinematics(
                    link=franka.get_link("hand"),
                    pos=np.array([[pos[0], pos[1], pos[2] + action.get("height", 0.18)]], dtype=np.float64),
                    quat=np.array([[0, 1, 0, 0]], dtype=np.float64),
                )
                if q.dim() == 2:
                    q = q[0]
                franka.control_dofs_position(q[:-2], np.arange(7))
                for _ in range(200):
                    self.scene.step()
                return {"ok": True, "result": True}
            return {"ok": False, "result": False}

        if act in ("wait", "settle"):
            self.scene.step(action.get("steps", 50))
            return {"ok": True, "result": True}

        return {"ok": False, "result": None,
                "reason": f"unknown action {act}"}

    def reset_objects(self):
        """Reset objects to original positions."""
        for name, orig_pos in self._original_positions.items():
            self.entities[name].set_pos(orig_pos, skip_forward=True)
        self.scene.step(10)

    def _is_done(self) -> torch.Tensor:
        """Check if current task is complete. Returns tensor."""
        if not hasattr(self, '_current_object') or not hasattr(self, '_current_target'):
            return torch.tensor(False, device=self.device)
        if self._current_object is None or self._current_target is None:
            return torch.tensor(False, device=self.device)

        obj = self.entities.get(self._current_object)
        if obj is None:
            return torch.tensor(False, device=self.device)

        obj_pos = obj.get_pos()
        target = torch.tensor(self._current_target, dtype=torch.float32, device=self.device)
        err = torch.norm(obj_pos[:2] - target[:2])
        # Object must be on ground (released) AND within 10cm of target
        on_ground = obj_pos[2] < 0.05
        return (err < 0.10) & on_ground

    def is_done(self) -> bool:
        """Check if current task is complete (for scripted control)."""
        result = self._is_done()
        if result.numel() == 1:
            return bool(result)
        return bool(result.all())

    # ── Camera ──────────────────────────────────────────────────

    def get_stereo_rgb(self, normalize=True) -> Optional[torch.Tensor]:
        """Get stereo RGB images. Returns ([B,] 6, H, W) or None."""
        if not self.left_cam:
            return None
        rgb_left = self.left_cam.read().rgb
        rgb_right = self.right_cam.read().rgb
        rgb_left = rgb_left.permute(0, 3, 1, 2).float()
        rgb_right = rgb_right.permute(0, 3, 1, 2).float()
        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0
        return torch.cat([rgb_left, rgb_right], dim=1)

    def render(self) -> np.ndarray:
        """Render visualization camera. Returns (H, W, 3) numpy array."""
        return self.vis_cam.render()[0]

    def start_recording(self):
        self.vis_cam.start_recording()

    def stop_recording(self, filename="output.mp4", fps=30):
        self.vis_cam.stop_recording(save_to_filename=filename, fps=fps)
