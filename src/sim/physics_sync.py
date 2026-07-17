"""
Module 13: Physics Sync
Step simulation and synchronize state.
"""
import time
from typing import Optional


class PhysicsSync:
    def __init__(self, scene, robot=None, objects=None):
        self.scene = scene
        self.robot = robot
        self.objects = objects or {}
        self.step_count = 0
        self.fps_history = []

    def step(self, n: int = 1) -> dict:
        start = time.time()
        for _ in range(n):
            self.scene.scene.step()
        elapsed = time.time() - start
        self.step_count += n

        if elapsed > 0:
            fps = n / elapsed
            self.fps_history.append(fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)

        return self.get_state()

    def get_state(self) -> dict:
        state = {"step": self.step_count}
        if self.robot:
            state["robot_qpos"] = self.robot.get_qpos().tolist()
            state["robot_pos"] = self.robot.get_pos().tolist()
        for name, obj in self.objects.items():
            state[f"object_{name}_pos"] = obj.get_pos().tolist()
        return state

    def get_fps(self) -> float:
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0

    def settle(self, steps: int = 100):
        self.step(steps)
