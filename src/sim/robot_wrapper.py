"""
Module 12: Robot Wrapper
Robot control interface for Genesis.
"""
import torch
from typing import Optional


class RobotWrapper:
    def __init__(self, entity):
        self.entity = entity
        self.n_dofs = entity.n_dofs

    def get_qpos(self) -> torch.Tensor:
        return self.entity.get_qpos()

    def set_qpos(self, qpos: torch.Tensor):
        self.entity.set_qpos(qpos)

    def get_pos(self) -> torch.Tensor:
        return self.entity.get_pos()

    def get_joint_positions(self) -> list:
        return self.get_qpos().tolist()

    def set_arm_joints(self, joints: list):
        qpos = self.get_qpos()
        for i, val in enumerate(joints[:7]):
            qpos[i] = val
        self.set_qpos(qpos)

    def set_gripper(self, width: float):
        qpos = self.get_qpos()
        if self.n_dofs > 7:
            qpos[7] = width
        self.set_qpos(qpos)

    def open_gripper(self):
        self.set_gripper(0.04)

    def close_gripper(self):
        self.set_gripper(0.0)

    def move_to_joints(self, joints: list, gripper: Optional[float] = None, steps: int = 100):
        qpos = self.get_qpos()
        for i, val in enumerate(joints[:7]):
            qpos[i] = val
        if gripper is not None and self.n_dofs > 7:
            qpos[7] = gripper
        self.set_qpos(qpos)
