"""
Robot Primitives — Comprehensive P0+P1 Fixes

All 38 issues addressed where possible in this module.
"""
import numpy as np
import torch
import logging
from typing import Dict, Optional, Tuple, List

# P2-35: Structured logging
logger = logging.getLogger("robopilot.primitives")


class RobotPrimitives:
    """Robot control primitives with all Genesis best practices."""

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

        # P0-06: Action scaling
        self.action_scale = 0.05

        # P0-04: Episode state
        self.episode_step = 0
        self.max_episode_steps = 500
        self.done = False

        # P1-16: Contact sensors
        self.contact_sensor = None

        # PD gains
        robot.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]),
                          dofs_idx_local=list(range(self.n_dofs)))
        robot.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]),
                          dofs_idx_local=list(range(self.n_dofs)))

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

    def _q_limit_tensors(self):
        lims = torch.tensor(np.asarray(self.robot.q_limit), dtype=torch.float32,
                            device=self.robot.get_qpos().device)
        return lims[0], lims[1]

    # ── P0-01: IK with FK Verification ─────────────────────

    TOP_DOWN_QUAT = np.array([0, 1, 0, 0])

    def solve_ik(self, x, y, z, quat=None, verify=True):
        if quat is None:
            quat = self.TOP_DOWN_QUAT
        qpos = self.robot.inverse_kinematics(
            link=self.ee_link, pos=np.array([x, y, z]), quat=np.array(quat))

        if verify:
            self.robot.set_qpos(qpos, list(range(self.n_dofs)))
            self.scene.step(5)
            actual = self._hand_pos()
            error = np.linalg.norm(actual - np.array([x, y, z]))
            if error > 0.05:
                logger.warning(f"IK error {error*100:.1f}cm")
                return None
        return qpos.tolist()

    # ── P0-02: PD Closed-Loop ──────────────────────────────

    def pd_move_to_xyz(self, target_xyz, steps=300, gripper_width=0.04):
        device = self.robot.get_qpos().device
        lo, hi = self._q_limit_tensors()

        joints = self.solve_ik(target_xyz[0], target_xyz[1], target_xyz[2])
        if joints is None:
            logger.error("IK failed, skipping move")
            return
        arm_target = torch.clamp(
            torch.tensor(joints[:self.n_arm], dtype=torch.float32, device=device),
            lo[:self.n_arm], hi[:self.n_arm])
        grip_target = torch.tensor(
            [gripper_width] * len(self.gripper_dofs), dtype=torch.float32, device=device)

        for _ in range(steps):
            self.robot.control_dofs_position(arm_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(grip_target, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

    # ── P0-04: Episode Structure ────────────────────────────

    def reset(self):
        self.episode_step = 0
        self.done = False
        home = np.concatenate([np.zeros(self.n_arm), [0.04, 0.04]])
        self.robot.set_qpos(home, list(range(self.n_dofs)))
        self.scene.step(100)
        return self._get_obs()

    def _get_obs(self):
        hand = self._hand_pos()
        qpos = self._to_numpy(self.robot.get_qpos())
        return np.concatenate([hand, qpos[:self.n_arm]])

    def is_done(self):
        return self.done or self.episode_step >= self.max_episode_steps

    # ── P0-05 & P0-06: Delta Action + Scaling ──────────────

    def apply_delta_action(self, action, dt=0.01):
        scaled = action * self.action_scale
        hand_pos = self._hand_pos()
        target = hand_pos + scaled[:3]
        target_joints = self.solve_ik(target[0], target[1], target[2], verify=False)

        device = self.robot.get_qpos().device
        lo, hi = self._q_limit_tensors()
        arm_target = torch.clamp(
            torch.tensor(target_joints[:self.n_arm], dtype=torch.float32, device=device),
            lo[:self.n_arm], hi[:self.n_arm])
        self.robot.control_dofs_position(arm_target, dofs_idx_local=self.arm_dofs)
        self.scene.step(1)
        self.episode_step += 1
        return self._get_obs()

    # ── P0-03: Weld + Contact Verify ───────────────────────

    def suction_grasp(self, obj_name):
        obj = self.objects[obj_name]
        obj_pos = self._to_numpy(obj.get_pos())
        obj_solver_idx = obj.link_start

        self.pd_move_to_xyz([obj_pos[0], obj_pos[1], obj_pos[2] + 0.05], steps=200)

        rigid = self.scene.rigid_solver
        link_obj = np.array([obj_solver_idx], dtype=np.int32)
        link_hand = np.array([self.ee_link.idx], dtype=np.int32)
        rigid.add_weld_constraint(link_obj, link_hand)
        self.scene.step(50)

        # P1-16: Contact verification
        contacts = obj.get_contacts()
        if contacts:
            forces = contacts.get("force_a")
            if forces is not None:
                f = self._to_numpy(forces)
                total = np.linalg.norm(f)
                return total > 0.1, total
        return False, 0.0

    def suction_release(self, obj_name):
        obj = self.objects[obj_name]
        rigid = self.scene.rigid_solver
        link_obj = np.array([obj.link_start], dtype=np.int32)
        link_hand = np.array([self.ee_link.idx], dtype=np.int32)
        rigid.delete_weld_constraint(link_obj, link_hand)
        self.scene.step(50)

    def open_gripper(self, width=0.04, steps=100):
        target = torch.tensor([width] * len(self.gripper_dofs),
                              dtype=torch.float32, device=self.robot.get_qpos().device)
        self.robot.control_dofs_position(target, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)
