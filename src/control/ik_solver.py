"""
Module 8: Standalone IK Solver
NOTE: The runtime path (primitives.py, orchestrator_v2) uses an inline IK
implementation in RobotPrimitives._solve_ik with state-safe FK. This module
is kept as a standalone reference but is not currently imported.
"""
import numpy as np
from typing import List, Tuple, Optional


class IKSolver:
    def __init__(self, robot_entity):
        self.robot = robot_entity
        self.n_joints = 7

    def solve(self, target_pos: List[float], target_rot: Optional[List[float]] = None) -> List[float]:
        current = self.robot.get_qpos().tolist()[:self.n_joints]
        target = np.array(target_pos[:3])

        for iteration in range(50):
            ee_pos = self._forward_kinematics(current)
            error = target - np.array(ee_pos[:3])

            if np.linalg.norm(error) < 0.001:
                break

            jacobian = self._compute_jacobian(current)
            jacobian_pos = jacobian[:3, :self.n_joints]

            try:
                delta = np.linalg.lstsq(jacobian_pos, error, rcond=None)[0]
            except np.linalg.LinAlgError:
                delta = np.zeros(self.n_joints)

            delta = np.clip(delta, -0.1, 0.1)
            current = [current[i] + delta[i] for i in range(self.n_joints)]

        return current

    def solve_above(self, object_pos: List[float], height: float = 0.15) -> List[float]:
        target = [object_pos[0], object_pos[1], object_pos[2] + height]
        return self.solve(target)

    def _forward_kinematics(self, joints: List[float]) -> List[float]:
        qpos = self.robot.get_qpos().tolist()
        for i, val in enumerate(joints[:self.n_joints]):
            qpos[i] = val
        self.robot.set_qpos(qpos)
        pos = self.robot.get_pos().tolist()
        return pos + [0, 0, 0]

    def _compute_jacobian(self, joints: List[float]) -> np.ndarray:
        eps = 0.01
        jacobian = np.zeros((6, self.n_joints))
        qpos_base = self.robot.get_qpos().tolist()

        for i in range(self.n_joints):
            qpos_plus = qpos_base.copy()
            qpos_plus[i] += eps
            self.robot.set_qpos(qpos_plus)
            pos_plus = self.robot.get_pos().tolist()

            qpos_minus = qpos_base.copy()
            qpos_minus[i] -= eps
            self.robot.set_qpos(qpos_minus)
            pos_minus = self.robot.get_pos().tolist()

            for j in range(3):
                jacobian[j, i] = (pos_plus[j] - pos_minus[j]) / (2 * eps)

        self.robot.set_qpos(qpos_base)
        return jacobian
