"""
Robot Primitives — OMPL approach + teleport grasp for pick-and-place.

Strategy:
1. plan_path (OMPL) to approach above object (collision-free)
2. set_qpos teleport to grasp height (minimal displacement)
3. control_dofs_position gripper close (gradual)
4. set_qpos teleport to lift

Objects should be on a Kinematic table for stability.
Uses standard panda.xml (works with OMPL plan_path).
"""
import numpy as np
import torch
from typing import List, Optional


class RobotPrimitives:
    def __init__(self, robot, scene, objects=None):
        self.robot = robot
        self.scene = scene
        self.objects = objects or {}
        self.n_dofs = robot.n_dofs
        self.n_arm = min(7, self.n_dofs)
        self.gripper_dofs = list(range(self.n_arm, self.n_dofs))
        self.arm_dofs = list(range(self.n_arm))

        self.ee_link = self._find_link("hand", "link7", "panda_hand", "panda_link7")
        self.ee_link_idx = self._find_link_list_index(self.ee_link)

        robot.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local=list(range(self.n_dofs)),
        )
        robot.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            dofs_idx_local=list(range(self.n_dofs)),
        )
        robot.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=list(range(self.n_dofs)),
        )

    def _find_link(self, *candidates):
        names = {ln.name: ln for ln in self.robot.links}
        for c in candidates:
            if c in names:
                return names[c]
        return list(names.values())[-1]

    def _find_link_list_index(self, link):
        for i, ln in enumerate(self.robot.links):
            if ln is link:
                return i
        return len(self.robot.links) - 1

    def _to_numpy(self, val):
        if hasattr(val, 'cpu'):
            return val.cpu().numpy()
        return np.asarray(val)

    # ── IK ───────────────────────────────────────────────────

    TOP_DOWN_QUAT = np.array([0, 1, 0, 0])

    def solve_ik(self, x, y, z, quat=None):
        if quat is None:
            quat = self.TOP_DOWN_QUAT
        qpos = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=np.array([x, y, z]),
            quat=np.array(quat),
        )
        return qpos.tolist()

    # ── Plan & Execute ───────────────────────────────────────

    def plan_and_execute(self, target_xyz, gripper_width=0.04, steps_per_wp=3):
        """Plan collision-free path and execute. Returns True if path is valid."""
        device = self.robot.get_qpos().device
        qpos = self.solve_ik(target_xyz[0], target_xyz[1], target_xyz[2])
        target_arr = np.array(qpos)
        target_arr[-2:] = gripper_width

        path = self.robot.plan_path(qpos_goal=target_arr, num_waypoints=200)

        first = np.array(path[0].cpu().numpy()[:self.n_dofs])
        last = np.array(path[-1].cpu().numpy()[:self.n_dofs])
        if np.abs(last - first).max() < 1e-5:
            return False

        for wp in path:
            wp_np = wp.cpu().numpy()[:self.n_dofs]
            wp_tensor = torch.tensor(wp_np, dtype=torch.float32, device=device)
            self.robot.control_dofs_position(wp_tensor)
            self.scene.step(steps_per_wp)
        return True

    # ── Teleport ─────────────────────────────────────────────

    def teleport(self, target_xyz, gripper_width=0.04, settle_steps=50):
        """Teleport arm to target XYZ via IK + set_qpos."""
        qpos = self.solve_ik(target_xyz[0], target_xyz[1], target_xyz[2])
        qpos_arr = np.array(qpos)
        qpos_arr[-2:] = gripper_width
        self.robot.set_qpos(qpos_arr, list(range(self.n_dofs)))
        self.scene.step(settle_steps)

    # ── Open Gripper ─────────────────────────────────────────

    def open_gripper(self, width=0.04, steps=100):
        target = torch.tensor([width] * len(self.gripper_dofs),
                              dtype=torch.float32, device=self.robot.get_qpos().device)
        self.robot.control_dofs_position(target, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)

    # ── Pick ─────────────────────────────────────────────────

    def pick(self, object_pos, grasp_force=1.5, steps=500):
        """Pick: plan approach → teleport grasp → close gripper → teleport lift."""
        op = self._to_numpy(object_pos)

        # Open gripper
        self.open_gripper(0.04, steps=100)

        # OMPL approach above cup
        above = [op[0], op[1], op[2] + 0.15]
        planned = self.plan_and_execute(above, gripper_width=0.04)

        # Re-read cup position
        cup_now = self._to_numpy(self.objects['red_cup'].get_pos()) if 'red_cup' in self.objects else op

        # Teleport to grasp height
        self.teleport([cup_now[0], cup_now[1], cup_now[2] + 0.05],
                      gripper_width=0.04, settle_steps=100)

        # Close gripper via control_dofs_position (gradual)
        device = self.robot.get_qpos().device
        current_arm = self.robot.get_qpos()[:self.n_arm].clone().to(device)
        for i in range(steps):
            t = (i + 1) / steps
            gw = 0.04 - t * (0.04 - 0.032)
            self.robot.control_dofs_position(current_arm, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(
                torch.tensor([gw, gw], dtype=torch.float32, device=device),
                dofs_idx_local=self.gripper_dofs,
            )
            self.scene.step(1)

        # Force hold
        force = torch.tensor([-grasp_force, -grasp_force],
                             dtype=torch.float32, device=device)
        for _ in range(100):
            self.robot.control_dofs_position(current_arm, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_force(force, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

        # Teleport lift
        cup_final = self._to_numpy(self.objects['red_cup'].get_pos()) if 'red_cup' in self.objects else cup_now
        self.teleport([cup_final[0], cup_final[1], cup_final[2] + 0.20],
                      gripper_width=0.032, settle_steps=200)

    # ── Place ────────────────────────────────────────────────

    def place(self, target_pos, steps=500):
        """Place: teleport above → teleport descend → open gripper → teleport lift."""
        tp = self._to_numpy(target_pos)

        self.teleport([tp[0], tp[1], tp[2] + 0.12], gripper_width=0.032)
        self.teleport([tp[0], tp[1], tp[2] + 0.02], gripper_width=0.032)
        self.open_gripper(0.04, steps=steps)
        self.teleport([tp[0], tp[1], tp[2] + 0.15], gripper_width=0.04)

    # ── Pick & Place ─────────────────────────────────────────

    def pick_and_place(self, object_pos, target_pos):
        self.pick(object_pos)
        self.place(target_pos)
