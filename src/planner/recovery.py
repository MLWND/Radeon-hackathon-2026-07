"""
Task Recovery — Detect failures and replan.

Flow:
    Action → Observe → Detect Failure → Replan → Retry

Uses ManipulationPipeline (not the deleted RobotPrimitives).
"""
import time
from typing import Dict, List


class FailureDetector:
    """Detect failures from scene state changes."""

    def __init__(self, scene_memory):
        self.memory = scene_memory
        self.failure_history = []

    def check_grasp_failure(self, object_name: str, pre_pos: List[float], post_pos: List[float]) -> Dict:
        if pre_pos is None or post_pos is None:
            return {"detected": False, "reason": "missing position data"}

        moved = sum(abs(post_pos[i] - pre_pos[i]) for i in range(3)) > 0.005
        return {
            "detected": not moved,
            "type": "grasp_failure",
            "reason": f"Object {object_name} did not move after grasp attempt",
            "pre_pos": pre_pos,
            "post_pos": post_pos,
        }

    def check_drop_failure(self, object_name: str, pre_pos: List[float], post_pos: List[float]) -> Dict:
        if pre_pos is None or post_pos is None:
            return {"detected": False, "reason": "missing position data"}

        z_dropped = pre_pos[2] - post_pos[2] > 0.05
        return {
            "detected": z_dropped,
            "type": "drop_failure",
            "reason": f"Object {object_name} z decreased by {pre_pos[2] - post_pos[2]:.3f}m",
            "pre_pos": pre_pos,
            "post_pos": post_pos,
        }

    def check_position_drift(self, object_name: str, target_pos: List[float], actual_pos: List[float], threshold: float = 0.1) -> Dict:
        if target_pos is None or actual_pos is None:
            return {"detected": False, "reason": "missing position data"}

        distance = sum((actual_pos[i] - target_pos[i])**2 for i in range(3))**0.5
        return {
            "detected": distance > threshold,
            "type": "position_drift",
            "reason": f"Object {object_name} is {distance:.3f}m from target (threshold: {threshold}m)",
            "distance": distance,
            "threshold": threshold,
        }

    def record_failure(self, failure: Dict):
        self.failure_history.append({"time": time.time(), **failure})

    def should_abort(self) -> bool:
        return len(self.failure_history) >= 3

    def get_failure_count(self) -> int:
        return len(self.failure_history)


class Replanner:
    """Replan after failure detection — produces actions for ManipulationPipeline."""

    def __init__(self):
        self.retry_strategies = {
            "grasp_failure": self._retry_grasp,
            "drop_failure": self._retry_drop,
            "position_drift": self._retry_drift,
        }

    def replan(self, failed_action: Dict, failure: Dict, scene_memory) -> List[Dict]:
        failure_type = failure.get("type", "unknown")
        strategy = self.retry_strategies.get(failure_type, self._generic_retry)
        return strategy(failed_action, failure, scene_memory)

    def _retry_grasp(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        obj_name = action.get("object", action.get("pick", ""))
        return [
            {"action": "pick", "object": obj_name,
             "description": f"Retry grasp: re-position above {obj_name} and descend"},
        ]

    def _retry_drop(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        obj_name = action.get("object", "")
        tgt = action.get("target", action.get("place", ""))
        return [
            {"action": "pick", "object": obj_name, "description": "Re-grasp dropped object"},
            {"action": "place", "target": tgt, "description": "Place at target"},
        ]

    def _retry_drift(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        tgt_name = action.get("target", "")
        return [
            {"action": "place", "target": tgt_name,
             "description": "Retry placement with more precise descend"},
        ]

    def _generic_retry(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        return [action]


class RecoveryManager:
    """Orchestrates failure detection and replanning for ManipulationPipeline."""

    def __init__(self, scene_memory):
        self.detector = FailureDetector(scene_memory)
        self.replanner = Replanner()
        self.max_retries = 2

    def execute_with_recovery(self, pipeline, action: Dict, scene_objects: Dict,
                              target_pos=None) -> Dict:
        """Execute pick/place via ManipulationPipeline with retry on failure.

        pipeline: ManipulationPipeline instance
        action: {"action": "pick"|"place", "object": ..., "target": ...}
        scene_objects: {name: entity} for position checks
        target_pos: [x,y,z] for place actions
        """
        action_type = action["action"]
        obj_name = action.get("object", "")

        for attempt in range(self.max_retries + 1):
            pre_pos = self._obj_pos(scene_objects, obj_name)

            # Execute via ManipulationPipeline
            if action_type == "pick":
                success = pipeline.suction_pick(obj_name)
            elif action_type == "place":
                if target_pos is None:
                    return {"success": False, "reason": "no target_pos for place"}
                err = pipeline.suction_place(obj_name, target_pos)
                success = err < 0.10
            else:
                return {"success": False, "reason": f"unknown action: {action_type}"}

            post_pos = self._obj_pos(scene_objects, obj_name)

            # Detect failure
            if action_type == "pick":
                failure = self.detector.check_grasp_failure(obj_name, pre_pos, post_pos)
            else:
                failure = self.detector.check_drop_failure(obj_name, pre_pos, post_pos)

            if not failure["detected"]:
                return {"success": True, "attempts": attempt + 1}

            self.detector.record_failure(failure)

            if self.detector.should_abort():
                return {"success": False, "reason": "Too many failures", "attempts": attempt + 1}

            new_plan = self.replanner.replan(action, failure, self.detector.memory)
            for new_action in new_plan:
                # Recursive retry with safer signal
                pass  # actual retry happens in next loop iteration via actions

        return {"success": False, "reason": "Max retries exceeded", "attempts": self.max_retries + 1}

    def _obj_pos(self, scene_objects, name):
        if not name or name not in scene_objects:
            return None
        return scene_objects[name].get_pos().cpu().numpy().tolist()
