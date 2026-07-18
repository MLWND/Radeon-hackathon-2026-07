"""
Robot Primitives — Suction gripper (weld constraint) pick-and-place.

Strategy:
1. plan_path (OMPL) to approach above object
2. Teleport to grasp height
3. Weld constraint (suction) → object attaches to hand
4. Teleport lift
5. Teleport to target
6. Unweld (release suction)

Objects: cubes/bottles on Kinematic table.
Uses standard panda.xml + OMPL plan_path.
"""
import numpy as np
import torch
from typing import Optional


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
        self.hand_solver_idx = self.robot.link_start + self.ee_link_idx

        robot.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local=list(range(self.n_dofs)),
        )
        robot.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
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
            self.robot.control_dofs_position(
                torch.tensor(wp_np, dtype=torch.float32, device=device))
            self.scene.step(steps_per_wp)
        return True

    # ── Teleport ─────────────────────────────────────────────

    def teleport(self, target_xyz, gripper_width=0.04, settle_steps=50):
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

    # ── Suction Pick ─────────────────────────────────────────

    def suction_pick(self, object_name):
        """Pick object using suction (weld constraint).

        Returns: True if object was lifted.
        """
        obj = self.objects[object_name]
        obj_pos = self._to_numpy(obj.get_pos())
        obj_solver_idx = obj.link_start  # first link of the object

        # Open gripper
        self.open_gripper(0.04, steps=100)

        # OMPL approach above object
        above = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.08]
        self.plan_and_execute(above, gripper_width=0.04)

        # Re-read object position
        obj_now = self._to_numpy(obj.get_pos())

        # Teleport to grasp height
        self.teleport([obj_now[0], obj_now[1], obj_now[2] + 0.02],
                      gripper_width=0.04, settle_steps=50)

        # Weld (suction)
        self.scene.rigid_solver.add_weld_constraint(
            self.hand_solver_idx, obj_solver_idx)
        self.scene.step(50)

        # Teleport lift
        obj_lifted = self._to_numpy(obj.get_pos())
        self.teleport([obj_lifted[0], obj_lifted[1], obj_lifted[2] + 0.15],
                      gripper_width=0.04, settle_steps=100)

        # Check if lifted
        final_z = self._to_numpy(obj.get_pos())[2]
        return final_z > obj_pos[2] + 0.03

    # ── Suction Place ────────────────────────────────────────

    def suction_place(self, object_name, target_pos):
        """Place object at target using suction release."""
        obj = self.objects[object_name]
        tp = self._to_numpy(target_pos)
        obj_solver_idx = obj.link_start

        # Approach above target
        self.teleport([tp[0], tp[1], tp[2] + 0.15],
                      gripper_width=0.04, settle_steps=50)

        # Descend to target
        self.teleport([tp[0], tp[1], tp[2] + 0.02],
                      gripper_width=0.04, settle_steps=50)

        # Unweld (release suction)
        self.scene.rigid_solver.delete_weld_constraint(
            self.hand_solver_idx, obj_solver_idx)
        self.scene.step(50)

        # Lift away
        self.teleport([tp[0], tp[1], tp[2] + 0.15],
                      gripper_width=0.04, settle_steps=50)

    # ── Pick & Place ─────────────────────────────────────────

    def pick_and_place(self, object_name, target_pos):
        """Full suction pick and place."""
        lifted = self.suction_pick(object_name)
        if lifted:
            self.suction_place(object_name, target_pos)
        return lifted

    # ── Legacy interface (for orchestrator) ──────────────────

    def pick(self, object_pos, grasp_force=1.5, steps=500):
        """Legacy pick — uses first object in scene."""
        if self.objects:
            name = list(self.objects.keys())[0]
            self.suction_pick(name)

    def place(self, target_pos, steps=500):
        """Legacy place — uses first object in scene."""
        if self.objects:
            name = list(self.objects.keys())[0]
            self.suction_place(name, target_pos)
