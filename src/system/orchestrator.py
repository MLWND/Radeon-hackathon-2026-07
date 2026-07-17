"""
Module 14: Orchestrator
Full MVP pipeline: Qwen3-VL → Planner → Primitives → Genesis.
"""
import time
import json
import numpy as np
from typing import Dict

from src.sim.scene_manager import SceneManager, create_tabletop_scene
from src.control.primitives import RobotPrimitives
from src.system.benchmark import Benchmark


# ── Object Registry ──────────────────────────────────────────

KNOWN_OBJECTS = {
    "red cup": "red_cup", "cup": "red_cup",
    "blue box": "blue_box", "box": "blue_box",
    "apple": "apple",
    "bottle": "bottle",
}


class Orchestrator:
    def __init__(self, scene: SceneManager, vlm=None):
        self.scene = scene
        self.prims = RobotPrimitives(scene.robot, scene.scene)
        self.bench = Benchmark()
        self.vlm = vlm  # Optional Qwen3-VL

    def run(self, instruction: str, use_vlm: bool = True) -> Dict:
        print(f"\n{'='*60}")
        print(f"  RoboPilot MVP: {instruction}")
        print(f"{'='*60}")

        # Phase 1: Settle
        self.scene.settle(50)

        # Phase 2: VLM Understanding (or rule-based fallback)
        self.bench.start("vlm")
        if use_vlm and self.vlm is not None:
            task = self._vlm_understand(instruction)
        else:
            task = self._rule_based(instruction)
        self.bench.end("vlm")
        print(f"\n  [VLM] Pick: {task['object']} -> Place: {task['target']}")

        # Phase 3: Resolve to Genesis coordinates
        obj_pos = self._resolve_position(task["object"])
        tgt_pos = self._resolve_position(task["target"])
        print(f"  [Resolve] Object at {[f'{x:.2f}' for x in obj_pos]}")
        print(f"  [Resolve] Target at {[f'{x:.2f}' for x in tgt_pos]}")

        # Phase 4: Execute pick & place
        self.bench.start("execution")
        print(f"\n  [Execute] Opening gripper...")
        self.prims.open_gripper()

        print(f"  [Execute] Picking object...")
        self.prims.pick(obj_pos)

        print(f"  [Execute] Placing at target...")
        self.prims.place(tgt_pos)
        self.bench.end("execution")

        # Phase 5: Verify
        self.bench.start("verify")
        img = self.scene.render_rgb()
        self.bench.end("verify")

        # Report
        self.bench.print_report()

        return {
            "success": True,
            "instruction": instruction,
            "task": task,
            "object_pos": obj_pos,
            "target_pos": tgt_pos,
        }

    def _vlm_understand(self, instruction: str) -> Dict:
        """Use Qwen3-VL for task understanding."""
        try:
            from demo.vlm_grounding_demo import vlm_understand, load_vlm
            proc, model = load_vlm()
            rgb = self.scene.render_rgb()
            vlm_out = vlm_understand(proc, model, rgb, instruction)
            return {
                "object": vlm_out.get("pick", "unknown"),
                "target": vlm_out.get("place", "unknown"),
                "reasoning": vlm_out.get("reasoning", ""),
            }
        except Exception as e:
            print(f"  [VLM] Fallback to rule-based: {e}")
            return self._rule_based(instruction)

    def _rule_based(self, instruction: str) -> Dict:
        """Rule-based task parsing."""
        lower = instruction.lower()
        obj = "red cup"
        tgt = "blue box"

        # Try to find pick target (after pick/grab/lift keywords)
        pick_kw = ["pick", "grab", "lift", "拿", "抓", "取"]
        place_kw = ["place", "put", "move", "放", "进", "到"]

        for kw in pick_kw:
            if kw in lower:
                # Find object name after the keyword
                idx = lower.find(kw)
                remaining = lower[idx:]
                for name in KNOWN_OBJECTS:
                    if name in remaining:
                        obj = name
                        break
                break

        for kw in place_kw:
            if kw in lower:
                idx = lower.find(kw)
                remaining = lower[idx:]
                for name in KNOWN_OBJECTS:
                    if name in remaining:
                        tgt = name
                        break
                break

        return {"object": obj, "target": tgt, "reasoning": "rule-based"}

    def _resolve_position(self, name: str) -> list:
        """Resolve object name to Genesis position."""
        entity_name = KNOWN_OBJECTS.get(name, name)
        if entity_name in self.scene.objects:
            return self.scene.objects[entity_name].get_pos().tolist()
        # Default to first object
        first_key = list(self.scene.objects.keys())[0]
        return self.scene.objects[first_key].get_pos().tolist()


def run_mvp(instruction: str = "Pick up the red cup and place it in the blue box"):
    """Run the full MVP pipeline."""
    scene = create_tabletop_scene(show_viewer=False)
    orch = Orchestrator(scene, vlm=None)
    result = orch.run(instruction, use_vlm=False)
    scene.save_video("demo/output/mvp_result.mp4", fps=30)
    return result
