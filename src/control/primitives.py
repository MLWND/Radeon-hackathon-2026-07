"""
RoboPilot Manipulation Pipeline — Rebuilt from Official Genesis Examples

Follows:
- examples/manipulation/grasp_env.py (scene, robot, reset, episode)
- examples/rigid/suction_cup.py (suction grasp flow)
- examples/tutorials/IK_motion_planning_grasp.py (IK + path planning)

Key patterns from official:
1. Ground plane only (no kinematic table)
2. RigidOptions(box_box_detection=True)
3. plan_path for approach, control_dofs_position for execution
4. Weld constraint for suction grasp
5. control_dofs_position called ONCE per target (not every step) + convergence check
6. IK verified by FK — falls back to Jacobian DLS if FK error > 2cm
"""
import genesis as gs
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class ManipulationPipeline:
    """Rebuilt from official Genesis examples with Episode interface."""

    def __init__(self, robot, scene, entities: Dict, rigid_solver=None):
        self.robot = robot
        self.scene = scene
        self.entities = entities
        self.rigid_solver = rigid_solver or scene.sim.rigid_solver

        # Robot config (from official grasp_env.py)
        self.n_dofs = robot.n_dofs
        self.n_arm = 7
        self.motors_dof = np.arange(self.n_arm)
        self.fingers_dof = np.arange(self.n_arm, self.n_dofs)
        self.ee_link = robot.get_link("hand")
        self.ee_idx = self.ee_link.idx

        # PD gains (official tutorial)
        robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # Home pose (official grasp_env.py default)
        self.home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])

        # Track object states for reset
        self.original_positions = {}
        for name, ent in self.entities.items():
            self.original_positions[name] = self._to_numpy(ent.get_pos()).copy()

        # Episode state
        self._episode_started = False
        self._current_object = None
        self._current_target = None

    # ── Episode Interface (Gym-style) ──────────────────────────

    def reset(self):
        """Reset episode: robot to home, objects to original positions."""
        # Robot to home
        self.robot.set_qpos(self.home_qpos)
        # Objects to original positions
        for name, orig_pos in self.original_positions.items():
            if name in self.entities:
                pos_t = torch.tensor(orig_pos, dtype=torch.float32, device=gs.device)
                self.entities[name].set_pos(pos_t, skip_forward=True)
        self.scene.step(20)
        self._episode_started = True
        self._current_object = None
        self._current_target = None
        return self._get_obs()

    def _get_obs(self):
        """Return current observation (robot + object states)."""
        return {
            "robot_qpos": self._to_numpy(self.robot.get_qpos()),
            "ee_pos": self._to_numpy(self.robot.get_links_pos()[self.ee_idx]),
            "objects": {n: self._to_numpy(e.get_pos()) for n, e in self.entities.items()},
        }

    def is_done(self):
        """Episode done when object on ground near target."""
        if self._current_object is None or self._current_target is None:
            return False
        obj = self.entities[self._current_object]
        obj_pos = self._to_numpy(obj.get_pos())
        err = float(np.linalg.norm(obj_pos[:2] - np.array(self._current_target)[:2]))
        # Object must be on ground (released) AND within 10cm of target
        on_ground = obj_pos[2] < 0.05
        return err < 0.10 and on_ground

    # ── IK with FK Verification + Jacobian Fallback ────────────

    def solve_ik(self, x, y, z, tol=0.02):
        """Solve IK with FK verification. Fallback to Jacobian DLS on failure."""
        target = np.array([x, y, z])
        # Step 1: try Genesis built-in IK
        qpos = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=target,
            quat=np.array([0, 1, 0, 0]),
        )
        # Step 2: FK verify (non-destructive: saves+restores current qpos)
        fk_err = self._fk_verify(qpos, target)
        if fk_err < tol:
            return qpos
        # Step 3: Jacobian DLS fallback
        qpos_fb = self._jacobian_dls_ik(target, qpos)
        fk_err_fb = self._fk_verify(qpos_fb, target)
        return qpos_fb if fk_err_fb < fk_err else qpos

    def _fk_verify(self, qpos, target):
        """Compute FK error (without disturbing sim state). Saves+restores qpos."""
        cur = self._to_numpy(self.robot.get_qpos())
        self.robot.set_qpos(qpos)
        ee_pos = self._to_numpy(self.robot.get_links_pos()[self.ee_idx])
        self.robot.set_qpos(cur)  # restore immediately
        return float(np.linalg.norm(ee_pos - target))

    def _jacobian_dls_ik(self, target, qpos_init, max_iter=50, lam=0.01, tol=0.005):
        """Damped Least Squares IK via finite-difference Jacobian."""
        q = self._to_numpy(qpos_init).copy()[: self.n_arm]
        cur_full = self._to_numpy(self.robot.get_qpos())
        for _ in range(max_iter):
            cur_full[: self.n_arm] = q
            self.robot.set_qpos(cur_full)
            ee_pos = self._to_numpy(self.robot.get_links_pos()[self.ee_idx])
            err = target - ee_pos
            if np.linalg.norm(err) < tol:
                break
            # Finite-diff Jacobian (3x7 position only)
            J = np.zeros((3, self.n_arm))
            eps = 0.001
            for i in range(self.n_arm):
                qp = q.copy(); qp[i] += eps
                cur_full[: self.n_arm] = qp
                self.robot.set_qpos(cur_full)
                ep = self._to_numpy(self.robot.get_links_pos()[self.ee_idx])
                J[:, i] = (ep - ee_pos) / eps
            # DLS
            dq = J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(3)) @ err
            q = np.clip(q + dq, -np.pi, np.pi)
        cur_full[: self.n_arm] = q
        cur_full[-2:] = 0.04
        self.robot.set_qpos(cur_full)  # leave in solved state
        return cur_full

    # ── PD Hold with Convergence Check ─────────────────────────

    def _pd_hold_and_check(self, qpos_arm, steps, tol=0.03):
        """Hold target qpos for `steps`. Return (reached, final_ee_pos)."""
        self.robot.control_dofs_position(qpos_arm, self.motors_dof)
        for _ in range(steps):
            self.scene.step()
        ee_pos = self._to_numpy(self.robot.get_links_pos()[self.ee_idx])
        # qpos convergence check
        cur_q = self._to_numpy(self.robot.get_qpos())[: self.n_arm]
        reached = float(np.linalg.norm(cur_q - qpos_arm)) < tol
        return reached, ee_pos

    # ── Suction Pick (official suction_cup.py pattern) ─────────

    def suction_pick(self, obj_name):
        """Pick object using suction (weld constraint)."""
        obj = self.entities[obj_name]
        obj_pos = self._to_numpy(obj.get_pos())
        obj_link = obj.get_link("box_baselink")
        cube_link_idx = obj_link.idx
        hand_link_idx = self.ee_idx
        self._current_object = obj_name

        # 1. Open gripper
        self.robot.control_dofs_force(np.array([0.5, 0.5]), self.fingers_dof)
        self.scene.step(50)

        # 2. Plan path to pre-grasp
        qpos_above = self.solve_ik(obj_pos[0], obj_pos[1], obj_pos[2] + 0.25)
        qpos_above[-2:] = 0.04
        path = self.robot.plan_path(qpos_goal=qpos_above, num_waypoints=100)
        for waypoint in path:
            self.robot.control_dofs_position(waypoint)
            self.robot.control_dofs_force(np.array([0.5, 0.5]), self.fingers_dof)
            self.scene.step()
        for _ in range(100):
            self.scene.step()

        # 3. PD descent to grasp height (hold + check convergence)
        qpos_reach = self.solve_ik(obj_pos[0], obj_pos[1], obj_pos[2] + 0.05)
        reached, ee_pos = self._pd_hold_and_check(qpos_reach[:-2], 50)
        if not reached:
            # extra settle time
            for _ in range(50):
                self.scene.step()

        # 4. Weld (suction)
        self.rigid_solver.add_weld_constraint(cube_link_idx, hand_link_idx)

        # 5. Lift
        qpos_lift = self.solve_ik(obj_pos[0], obj_pos[1], obj_pos[2] + 0.28)
        self._pd_hold_and_check(qpos_lift[:-2], 50)

        final_z = float(obj.get_pos().cpu().numpy()[2])
        return final_z > obj_pos[2] + 0.05

    # ── Suction Place (official pattern) ───────────────────────

    def suction_place(self, obj_name, target_pos):
        """Place object at target using suction release."""
        obj = self.entities[obj_name]
        tp = self._to_numpy(target_pos)
        obj_link = obj.get_link("box_baselink")
        cube_link_idx = obj_link.idx
        hand_link_idx = self.ee_idx
        self._current_target = list(tp)

        # Reach to place height
        qpos_reach = self.solve_ik(tp[0], tp[1], tp[2] + 0.18)
        self._pd_hold_and_check(qpos_reach[:-2], 100)

        # Release + settle
        self.rigid_solver.delete_weld_constraint(cube_link_idx, hand_link_idx)
        for _ in range(400):
            self.scene.step()

        final = obj.get_pos().cpu().numpy()
        error = float(np.sqrt((final[0] - tp[0]) ** 2 + (final[1] - tp[1]) ** 2))
        return error

    # ── Stereo Camera (from official grasp_env.py) ─────────────

    def setup_stereo_cameras(self, image_size=(64, 64)):
        """Setup stereo cameras for VLM (official pattern)."""
        try:
            from genesis.options.sensors import RasterizerCameraOptions
            self.left_cam = self.scene.add_sensor(
                RasterizerCameraOptions(
                    res=(image_size[0], image_size[1]),
                    pos=(1.25, 0.3, 0.3),
                    lookat=(0.0, 0.0, 0.0),
                    fov=60,
                ))
            self.right_cam = self.scene.add_sensor(
                RasterizerCameraOptions(
                    res=(image_size[0], image_size[1]),
                    pos=(1.25, -0.3, 0.3),
                    lookat=(0.0, 0.0, 0.0),
                    fov=60,
                ))
        except Exception as e:
            self.left_cam = None
            self.right_cam = None

    def get_stereo_rgb(self, normalize=True):
        """Get stereo RGB images (official pattern)."""
        if not getattr(self, 'left_cam', None):
            return None
        rgb_left = self.left_cam.read().rgb
        rgb_right = self.right_cam.read().rgb
        rgb_left = rgb_left.permute(0, 3, 1, 2).float()
        rgb_right = rgb_right.permute(0, 3, 1, 2).float()
        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0
        return torch.cat([rgb_left, rgb_right], dim=1)

    # ── Action Dispatcher (driven by ActionScheduler) ──────────

    def execute_action(self, action: Dict, target_pos: Optional[list] = None):
        """Dispatch one action dict to the right primitive.

        This is the bridge from ActionScheduler to ManipulationPipeline.
        Returns per-action result; pick/place bool/float, None for unknown.
        """
        act = action.get("action", "").lower()
        obj = action.get("object", "")
        tgt = action.get("target", "")
        if act == "pick":
            if obj in self.entities:
                return {"ok": True, "result": self.suction_pick(obj)}
            return {"ok": False, "result": False, "reason": f"unknown object {obj}"}
        if act == "place":
            if obj == "":
                obj = self._current_object or list(self.entities.keys())[0]
            if obj not in self.entities:
                return {"ok": False, "result": 0.99, "reason": f"unknown object {obj}"}
            # Resolve target_pos: explicit, or from scene memory
            if target_pos is not None:
                tp = target_pos
            elif tgt in self.entities:
                tp = self._to_numpy(self.entities[tgt].get_pos()).tolist()
            else:
                tp = [0.55, 0.0, 0.02]
            return {"ok": True, "result": self.suction_place(obj, tp)}
        if act == "move_above":
            name = obj or tgt
            if name in self.entities:
                pos = self._to_numpy(self.entities[name].get_pos())
                q = self.solve_ik(pos[0], pos[1], pos[2] + action.get("height", 0.18))
                self._pd_hold_and_check(q[:-2], 50)
                return {"ok": True, "result": True}
            return {"ok": False, "result": False}
        if act == "wait" or act == "settle":
            self.scene.step(action.get("steps", 50))
            return {"ok": True, "result": True}
        return {"ok": False, "result": None, "reason": f"unknown action {act}"}

    # ── Object Reset ───────────────────────────────────────────

    def reset_objects(self):
        """Reset objects to original positions."""
        for name, orig_pos in self.original_positions.items():
            if name in self.entities:
                pos_tensor = torch.tensor(orig_pos, dtype=torch.float32, device=gs.device)
                self.entities[name].set_pos(pos_tensor, skip_forward=True)
        self.scene.step(10)

    # ── Helper ─────────────────────────────────────────────────

    def _to_numpy(self, val):
        if hasattr(val, 'cpu'):
            return val.cpu().numpy()
        return np.asarray(val)
