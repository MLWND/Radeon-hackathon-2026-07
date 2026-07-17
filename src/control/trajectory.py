"""
Module 9: Trajectory Generator
Smooth trajectory generation between waypoints.
"""
import numpy as np
from typing import List


class TrajectoryGenerator:
    def __init__(self, steps_per_waypoint: int = 50):
        self.steps = steps_per_waypoint

    def linear(self, start: List[float], end: List[float]) -> List[List[float]]:
        trajectory = []
        for i in range(self.steps):
            t = i / (self.steps - 1)
            point = [start[j] + (end[j] - start[j]) * t for j in range(len(start))]
            trajectory.append(point)
        return trajectory

    def smooth(self, waypoints: List[List[float]]) -> List[List[float]]:
        if len(waypoints) < 2:
            return waypoints

        trajectory = []
        for i in range(len(waypoints) - 1):
            segment = self.linear(waypoints[i], waypoints[i + 1])
            trajectory.extend(segment[:-1])
        trajectory.append(waypoints[-1])
        return trajectory

    def arc(self, center: List[float], radius: float, start_angle: float, end_angle: float, n_points: int = 50) -> List[List[float]]:
        trajectory = []
        for i in range(n_points):
            t = i / (n_points - 1)
            angle = start_angle + (end_angle - start_angle) * t
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            trajectory.append([x, y, z])
        return trajectory
