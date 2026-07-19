"""
RoboPilot Manipulation Pipeline — Rebuilt from Official Genesis Examples

Follows:
- examples/manipulation/grasp_env.py (scene, robot, reset)
- examples/rigid/suction_cup.py (suction grasp flow)
- examples/tutorials/IK_motion_planning_grasp.py (IK + path planning)

Key patterns from official:
1. Ground plane only (no kinematic table)
2. RigidOptions(box_box_detection=True)
3. plan_path for approach, control_dofs_position for execution
4. Weld constraint for suction grasp
5. control_dofs_position called ONCE per target (not every step)
"""
import genesis as gs
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class ManipulationPipeline:
    """Rebuilt from official Genesis examples."""

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

        # PD gains (from official examples)
        robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # Track object states for reset
        self.original_positions = {}
        for name, ent in self.entities.items():
            self.original_positions[name] = self._to_numpy(ent.get_pos()).copy()

    # ── Stereo Camera (from official grasp_env.py) ──────────

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
            print(f"  Stereo cameras: {image_size[0]}x{image_size[1]}")
        except Exception as e:
            print(f"  Stereo camera setup failed: {e}")

    def get_stereo_rgb(self, normalize=True):
        """Get stereo RGB images (official pattern)."""
        if not hasattr(self, 'left_cam'):
            return None
        rgb_left = self.left_cam.read().rgb
        rgb_right = self.right_cam.read().rgb
        rgb_left = rgb_left.permute(0, 3, 1, 2).float()
        rgb_right = rgb_right.permute(0, 3, 1, 2).float()
        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0
        return torch.cat([rgb_left, rgb_right], dim=1)

    # ── IK (official pattern) ───────────────────────────────

    def solve_ik(self, x, y, z):
        """Solve IK for target position (top-down orientation)."""
        return self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=np.array([x, y, z]),
            quat=np.array([0, 1, 0, 0]),
        )

    # ── Suction Pick (official suction_cup.py pattern) ──────

    def suction_pick(self, obj_name):
        """Pick object using suction (weld constraint).

        Official flow: plan_path → reach → weld → lift
        """
        obj = self.entities[obj_name]
        obj_pos = self._to_numpy(obj.get_pos())
        obj_link = obj.get_link("box_baselink")
        cube_link_idx = obj_link.idx
        hand_link_idx = self.ee_link.idx

        # 1. Open gripper
        self.robot.control_dofs_force(np.array([0.5, 0.5]), self.fingers_dof)
        self.scene.step(50)

        # 2. Plan path to pre-grasp (above cube)
        qpos_above = self.solve_ik(obj_pos[0], obj_pos[1], obj_pos[2] + 0.25)
        qpos_above[-2:] = 0.04  # open gripper
        path = self.robot.plan_path(qpos_goal=qpos_above, num_waypoints=100)

        for waypoint in path:
            self.robot.control_dofs_position(waypoint)
            self.robot.control_dofs_force(np.array([0.5, 0.5]), self.fingers_dof)
            self.scene.step()

        # Settle
        for _ in range(100):
            self.scene.step()

        # 3. Reach to grasp height
        qpos_reach = self.solve_ik(obj_pos[0], obj_pos[1], obj_pos[2] + 0.05)
        self.robot.control_dofs_position(qpos_reach[:-2], self.motors_dof)
        for _ in range(50):
            self.scene.step()

        # 4. Weld (suction)
        self.rigid_solver.add_weld_constraint(cube_link_idx, hand_link_idx)

        # 5. Lift
        qpos_lift = self.solve_ik(obj_pos[0], obj_pos[1], obj_pos[2] + 0.28)
        self.robot.control_dofs_position(qpos_lift[:-2], self.motors_dof)
        for _ in range(50):
            self.scene.step()

        # Check if lifted
        final_z = obj.get_pos().cpu().numpy()[2]
        return final_z > obj_pos[2] + 0.05

    # ── Suction Place (official pattern) ────────────────────

    def suction_place(self, obj_name, target_pos):
        """Place object at target using suction release."""
        obj = self.entities[obj_name]
        tp = self._to_numpy(target_pos)
        obj_link = obj.get_link("box_baselink")
        cube_link_idx = obj_link.idx
        hand_link_idx = self.ee_link.idx

        # Reach to place position
        qpos_reach = self.solve_ik(tp[0], tp[1], tp[2] + 0.18)
        self.robot.control_dofs_position(qpos_reach[:-2], self.motors_dof)
        for _ in range(100):
            self.scene.step()

        # Release
        self.rigid_solver.delete_weld_constraint(cube_link_idx, hand_link_idx)
        for _ in range(400):
            self.scene.step()

        # Check placement
        final = obj.get_pos().cpu().numpy()
        error = np.sqrt((final[0] - tp[0])**2 + (final[1] - tp[1])**2)
        return error

    # ── Reset (from official grasp_env.py) ──────────────────

    def reset_objects(self):
        """Reset objects to original positions (official pattern)."""
        for name, orig_pos in self.original_positions.items():
            if name in self.entities:
                pos_tensor = torch.tensor(orig_pos, dtype=torch.float32, device=gs.device)
                self.entities[name].set_pos(pos_tensor, skip_forward=True)
        self.scene.step(10)

    # ── Helper ──────────────────────────────────────────────

    def _to_numpy(self, val):
        if hasattr(val, 'cpu'):
            return val.cpu().numpy()
        return np.asarray(val)
