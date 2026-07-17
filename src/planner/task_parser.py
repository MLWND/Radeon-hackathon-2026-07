"""
Module 5: Task Parser
Parse VLM output into action sequence.
"""
from typing import Dict, List


class TaskParser:
    def __init__(self):
        self.action_map = {
            "pick_place": self._parse_pick_place,
            "stack": self._parse_stack,
            "sort": self._parse_sort,
        }

    def parse(self, task_json: Dict) -> List[Dict]:
        task_type = task_json.get("task", "pick_place")
        parser = self.action_map.get(task_type, self._parse_pick_place)
        return parser(task_json)

    def _parse_pick_place(self, task: Dict) -> List[Dict]:
        obj = task.get("object", {})
        target = task.get("target", {})

        return [
            {"action": "move_to", "target": "above_object", "description": f"Move above {obj.get('color', '')} {obj.get('type', 'object')}"},
            {"action": "move_down", "target": "object", "description": "Lower to object"},
            {"action": "close_gripper", "description": "Grasp object"},
            {"action": "lift", "height": 0.15, "description": "Lift object"},
            {"action": "move_to", "target": "above_target", "description": f"Move above {target.get('color', '')} {target.get('type', 'target')}"},
            {"action": "move_down", "target": "target", "description": "Lower to target"},
            {"action": "open_gripper", "description": "Release object"},
            {"action": "lift", "height": 0.15, "description": "Retract arm"},
        ]

    def _parse_stack(self, task: Dict) -> List[Dict]:
        return [
            {"action": "move_to", "target": "object"},
            {"action": "close_gripper"},
            {"action": "lift", "height": 0.2},
            {"action": "move_to", "target": "target"},
            {"action": "move_down", "target": "target"},
            {"action": "open_gripper"},
            {"action": "lift", "height": 0.1},
        ]

    def _parse_sort(self, task: Dict) -> List[Dict]:
        return self._parse_pick_place(task)
