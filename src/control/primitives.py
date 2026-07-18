"""
Robot Primitives — Suction gripper (weld constraint) pick-and-place.

Key insight: weld constraint breaks during teleport (set_qpos is too fast).
Fix: use PD control for place approach (gradual movement keeps weld intact).
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

    def _hand_pos(self):
        return self._to_numpy(self.robot.get_links_pos()[self.ee_link_idx])

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

    # ── Teleport (fast, for approach only) ───────────────────

    def teleport(self, target_xyz, gripper_width=0.04, settle_steps=50):
        qpos = self.solve_ik(target_xyz[0], target_xyz[1], target_xyz[2])
        qpos_arr = np.array(qpos)
        qpos_arr[-2:] = gripper_width
        self.robot.set_qpos(qpos_arr, list(range(self.n_dofs)))
        self.scene.step(settle_steps)

    # ── PD Move (gradual, for place with weld) ──────────────

    def pd_move(self, target_xyz, gripper_width=0.04, steps=300):
        """Gradual PD move — keeps weld constraint intact."""
        device = self.robot.get_qpos().device
        lo, hi = self._q_limit_tensors()

        joints = self.solve_ik(target_xyz[0], target_xyz[1], target_xyz[2])
        arm_target = torch.clamp(
            torch.tensor(joints[:self.n_arm], dtype=torch.float32, device=device),
            lo[:self.n_arm], hi[:self.n_arm])
        grip_target = torch.tensor(
            [gripper_width] * len(self.gripper_dofs), dtype=torch.float32, device=device)

        for _ in range(steps):
            self.robot.control_dofs_position(arm_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(grip_target, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

    def _q_limit_tensors(self):
        lims = torch.tensor(np.asarray(self.robot.q_limit), dtype=torch.float32,
                            device=self.robot.get_qpos().device)
        return lims[0], lims[1]

    # ── Open Gripper ─────────────────────────────────────────

    def open_gripper(self, width=0.04, steps=100):
        target = torch.tensor([width] * len(self.gripper_dofs),
                              dtype=torch.float32, device=self.robot.get_qpos().device)
        self.robot.control_dofs_position(target, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)

    # ── Suction Pick ─────────────────────────────────────────

    def suction_pick(self, object_name):
        obj = self.objects[object_name]
        obj_pos = self._to_numpy(obj.get_pos())
        obj_solver_idx = obj.link_start

        self.open_gripper(0.04, steps=100)

        # OMPL approach above object
        above = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.08]
        self.plan_and_execute(above, gripper_width=0.04)

        # Re-read position
        obj_now = self._to_numpy(obj.get_pos())

        # Teleport to grasp height
        self.teleport([obj_now[0], obj_now[1], obj_now[2] + 0.02],
                      gripper_width=0.04, settle_steps=50)

        # Weld
        self.scene.rigid_solver.add_weld_constraint(
            self.hand_solver_idx, obj_solver_idx)
        self.scene.step(50)

        # Teleport lift (weld holds because lift is short and vertical)
        obj_lifted = self._to_numpy(obj.get_pos())
        self.teleport([obj_lifted[0], obj_lifted[1], obj_lifted[2] + 0.15],
                      gripper_width=0.04, settle_steps=100)

        final_z = self._to_numpy(obj.get_pos())[2]
        return final_z > obj_pos[2] + 0.03

    # ── Suction Place (PD move keeps weld intact) ────────────

    def suction_place(self, object_name, target_pos):
        obj = self.objects[object_name]
        tp = self._to_numpy(target_pos)
        obj_solver_idx = obj.link_start

        # First: teleport to above target (fast, weld may slip a bit)
        self.teleport([tp[0], tp[1], tp[2] + 0.12],
                      gripper_width=0.04, settle_steps=50)

        # Then: PD descend to exact target (gradual, weld stays intact)
        self.pd_move([tp[0], tp[1], tp[2] + 0.02],
                     gripper_width=0.04, steps=300)

        # Verify hand is near target
        hand = self._hand_pos()
        hand_err = np.linalg.norm(hand[:2] - tp[:2])

        # Unweld
        self.scene.rigid_solver.delete_weld_constraint(
            self.hand_solver_idx, obj_solver_idx)
        self.scene.step(80)

        actual = self._to_numpy(obj.get_pos())
        error = np.linalg.norm(actual[:2] - tp[:2])

        # Lift away (teleport is fine after unweld)
        self.teleport([tp[0], tp[1], tp[2] + 0.15],
                      gripper_width=0.04, settle_steps=50)

        return error

    # ── Pick & Place ─────────────────────────────────────────

    def pick_and_place(self, object_name, target_pos):
        lifted = self.suction_pick(object_name)
        if lifted:
            error = self.suction_place(object_name, target_pos)
            return True, error
        return False, float('inf')

    # ── Legacy interface ─────────────────────────────────────

    def pick(self, object_pos, grasp_force=1.5, steps=500):
        if self.objects:
            self.suction_pick(list(self.objects.keys())[0])

    def place(self, target_pos, steps=500):
        if self.objects:
            self.suction_place(list(self.objects.keys())[0], target_pos)
