"""
Task Recovery — Detect failures and replan.

Flow:
    Action → Observe → Detect Failure → Replan → Retry

Supports:
    - Grasp failure detection
    - Object drop detection
    - Position drift detection
    - Automatic retry with adjusted strategy
"""
import time
from typing import Dict, List


class FailureDetector:
    """Detect failures from scene state changes."""

    def __init__(self, scene_memory):
        self.memory = scene_memory
        self.failure_history = []

    def check_grasp_failure(self, object_name: str, pre_pos: List[float], post_pos: List[float]) -> Dict:
        """Check if grasp failed (object didn't move with gripper)."""
        if pre_pos is None or post_pos is None:
            return {"detected": False, "reason": "missing position data"}

        moved = sum(abs(post_pos[i] - pre_pos[i]) for i in range(3)) > 0.005  # Lower threshold
        return {
            "detected": not moved,
            "type": "grasp_failure",
            "reason": f"Object {object_name} did not move after grasp attempt",
            "pre_pos": pre_pos,
            "post_pos": post_pos,
        }

    def check_drop_failure(self, object_name: str, pre_pos: List[float], post_pos: List[float]) -> Dict:
        """Check if object was dropped (z decreased significantly)."""
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
        """Check if object is too far from target."""
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
        self.failure_history.append({
            "time": time.time(),
            **failure,
        })

    def should_abort(self) -> bool:
        return len(self.failure_history) >= 3

    def get_failure_count(self) -> int:
        return len(self.failure_history)


class Replanner:
    """Replan after failure detection."""

    def __init__(self):
        self.retry_strategies = {
            "grasp_failure": self._retry_grasp,
            "drop_failure": self._retry_drop,
            "position_drift": self._retry_drift,
        }

    def replan(self, failed_action: Dict, failure: Dict, scene_memory) -> List[Dict]:
        """Generate new action plan after failure."""
        failure_type = failure.get("type", "unknown")
        strategy = self.retry_strategies.get(failure_type, self._generic_retry)
        return strategy(failed_action, failure, scene_memory)

    def _retry_grasp(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        """Retry grasp with adjusted approach."""
        obj_name = action.get("object", "")
        obj_pos = memory.get_position(obj_name)

        return [
            {"action": "open_gripper", "description": "Open gripper wide"},
            {"action": "move_above", "object": obj_name, "height": 0.15, "description": "Move higher above object"},
            {"action": "move_to", "object": obj_name, "offset_z": 0.02, "description": "Lower to object carefully"},
            {"action": "close_gripper", "description": "Close gripper firmly"},
            {"action": "lift", "object": obj_name, "height": 0.12, "description": "Lift object"},
        ]

    def _retry_drop(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        """Retry after drop — re-grasp and place."""
        obj_name = action.get("object", "")
        return [
            {"action": "move_to", "object": obj_name, "description": "Move to dropped object"},
            {"action": "close_gripper", "description": "Re-grasp object"},
            {"action": "lift", "object": obj_name, "height": 0.12, "description": "Lift object"},
            {"action": "place", "target": action.get("target", ""), "description": "Place at target"},
        ]

    def _retry_drift(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        """Retry placement with more precise approach."""
        return [
            {"action": "move_above", "target": action.get("target", ""), "height": 0.15, "description": "Move higher above target"},
            {"action": "place", "target": action.get("target", ""), "description": "Place carefully"},
        ]

    def _generic_retry(self, action: Dict, failure: Dict, memory) -> List[Dict]:
        return [action]


class RecoveryManager:
    """Orchestrates failure detection and replanning."""

    def __init__(self, scene_memory):
        self.detector = FailureDetector(scene_memory)
        self.replanner = Replanner()
        self.max_retries = 2

    def execute_with_recovery(self, primitives, action: Dict, scene_memory) -> Dict:
        """Execute action with automatic failure recovery."""
        obj_name = action.get("object", action.get("target", ""))

        for attempt in range(self.max_retries + 1):
            # Record pre-action state
            pre_pos = scene_memory.get_position(obj_name)

            # Execute action
            success = self._execute_action(primitives, action, scene_memory)

            # Record post-action state
            # CRITICAL FIX: need {name: entity} dict for update_positions.
            # primitives.objects is set by OrchestratorV2 from the SceneManager.
            scene_objects = getattr(primitives, 'objects', None) or {}
            scene_memory.update_positions(scene_objects)
            post_pos = scene_memory.get_position(obj_name)

            # Check for failure
            if action["action"] == "pick":
                failure = self.detector.check_grasp_failure(obj_name, pre_pos, post_pos)
            elif action["action"] == "place":
                failure = self.detector.check_drop_failure(obj_name, pre_pos, post_pos)
            else:
                failure = {"detected": False}

            if not failure["detected"]:
                return {"success": True, "attempts": attempt + 1}

            # Record failure
            self.detector.record_failure(failure)
            print(f"    [Recovery] Failure detected: {failure['reason']}")

            if self.detector.should_abort():
                return {"success": False, "reason": "Too many failures", "attempts": attempt + 1}

            # Replan
            new_plan = self.replanner.replan(action, failure, scene_memory)
            print(f"    [Recovery] Replanning: {len(new_plan)} new steps")

            # Execute new plan
            for new_action in new_plan:
                self._execute_action(primitives, new_action, scene_memory)

        return {"success": False, "reason": "Max retries exceeded", "attempts": self.max_retries + 1}

    def _execute_action(self, primitives, action: Dict, memory) -> bool:
        """Execute a single action."""
        action_type = action["action"]
        obj_name = action.get("object", "")
        tgt_name = action.get("target", "")

        if action_type == "pick":
            pos = memory.get_position(obj_name)
            if pos:
                primitives.pick(pos)
                return True
        elif action_type == "place":
            pos = memory.get_position(tgt_name)
            if pos:
                primitives.place(pos)
                return True
        elif action_type == "open_gripper":
            primitives.open_gripper()
            return True
        elif action_type == "close_gripper":
            primitives.close_gripper()
            return True
        elif action_type == "move_above":
            pos = memory.get_position(obj_name or tgt_name)
            if pos:
                height = action.get("height", 0.12)
                primitives.move_above_object(pos, height)
                return True
        elif action_type == "lift":
            pos = memory.get_position(obj_name)
            if pos:
                height = action.get("height", 0.12)
                primitives.move_above_object(pos, height)
                return True
        return False
