"""
Module 10: Standalone Gripper Controller
NOTE: Runtime path uses primitives.py _set_gripper inline.
Kept as a standalone reference module.
from enum import Enum


class GripperState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"


class GripperController:
    OPEN_WIDTH = 0.04
    CLOSED_WIDTH = 0.0
    CLOSE_FORCE = 20.0

    def __init__(self, robot_wrapper):
        self.robot = robot_wrapper
        self.state = GripperState.OPEN
        self.current_width = self.OPEN_WIDTH

    def open(self, steps: int = 30):
        self.state = GripperState.MOVING
        self.robot.set_gripper(self.OPEN_WIDTH)
        self.current_width = self.OPEN_WIDTH
        self.state = GripperState.OPEN

    def close(self, steps: int = 30):
        self.state = GripperState.MOVING
        self.robot.set_gripper(self.CLOSED_WIDTH)
        self.current_width = self.CLOSED_WIDTH
        self.state = GripperState.CLOSED

    def set_width(self, width: float):
        self.state = GripperState.MOVING
        self.robot.set_gripper(width)
        self.current_width = width
        self.state = GripperState.OPEN if width > 0.02 else GripperState.CLOSED

    def is_closed(self) -> bool:
        return self.state == GripperState.CLOSED

    def is_open(self) -> bool:
        return self.state == GripperState.OPEN
