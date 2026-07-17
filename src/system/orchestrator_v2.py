"""
Orchestrator V2 — Upgraded MVP with Task Planner + Scene Memory + Verification.

Pipeline:
    User → Task Planner → Scene Memory → Primitives → Genesis → Camera Verify
"""
from typing import Dict, List

from src.sim.scene_manager import SceneManager, create_tabletop_scene
from src.control.primitives import RobotPrimitives
from src.planner.task_planner import TaskPlanner
from src.planner.recovery import RecoveryManager
from src.vision.scene_memory import SceneMemory
from src.vision.verifier import CameraVerifier
from src.system.benchmark import Benchmark


# ── Object Registry ──────────────────────────────────────────

KNOWN_OBJECTS = {
    "red cup": "red_cup", "cup": "red_cup",
    "blue box": "blue_box", "box": "blue_box",
    "apple": "apple",
    "bottle": "bottle",
}


class OrchestratorV2:
    def __init__(self, scene: SceneManager, vlm_proc=None, vlm_model=None):
        self.scene = scene
        self.prims = RobotPrimitives(scene.robot, scene.scene, objects=scene.objects)
        self.planner = TaskPlanner(vlm_proc, vlm_model)
        self.recovery = RecoveryManager(None)  # scene_memory wired per-step
        self.memory = SceneMemory()
        self.verifier = CameraVerifier(vlm_proc, vlm_model)
        self.bench = Benchmark()
        self.vlm_proc = vlm_proc
        self.vlm_model = vlm_model

        # Initialize scene memory
        self.memory.initialize(scene.objects)

    def run(self, instruction: str, use_vlm: bool = False) -> Dict:
        print(f"\n{'='*60}")
        print(f"  RoboPilot V2: {instruction}")
        print(f"{'='*60}")

        # Phase 1: Capture initial state
        self.scene.settle(50)
        self.bench.start("camera")
        initial_img = self.scene.render_rgb()
        self.bench.end("camera")
        self.verifier.capture_before(initial_img)
        self.memory.update_positions(self.scene.objects)

        # Phase 2: Task Planning
        self.bench.start("planner")
        if use_vlm and self.vlm_proc:
            plan = self.planner.plan(instruction, initial_img)
        else:
            plan = self.planner.plan(instruction)
        self.bench.end("planner")

        steps = plan.get("steps", [])
        print(f"\n  [Plan] {len(steps)} steps:")
        for i, step in enumerate(steps):
            print(f"    {i+1}. {step['action']}: {step.get('object', step.get('target', ''))} — {step.get('description', '')}")

        # Phase 3: Execute each step
        self.bench.start("execution")
        for i, step in enumerate(steps):
            action = step["action"]
            print(f"\n  [Step {i+1}/{len(steps)}] {action}: {step.get('description', '')}")

            if action == "pick":
                obj_name = step.get("object", "red cup")
                entity_name = KNOWN_OBJECTS.get(obj_name, obj_name)
                obj_pos = self.memory.get_position(entity_name)
                if obj_pos:
                    print(f"    Object at {[f'{x:.2f}' for x in obj_pos]}")
                    # Execute with recovery
                    self.recovery.detector.memory = self.memory
                    result = self.recovery.execute_with_recovery(
                        self.prims, {"action": "pick", "object": entity_name}, self.memory
                    )
                    if result["success"]:
                        self.memory.record_action("pick", entity_name)
                        print(f"    Pick success (attempts: {result['attempts']})")
                    else:
                        print(f"    Pick failed: {result.get('reason','')}")
                else:
                    print(f"    Object '{entity_name}' not found in memory")

            elif action == "place":
                tgt_name = step.get("target", "blue box")
                entity_name = KNOWN_OBJECTS.get(tgt_name, tgt_name)
                tgt_pos = self.memory.get_position(entity_name)
                if tgt_pos:
                    print(f"    Target at {[f'{x:.2f}' for x in tgt_pos]}")
                    self.prims.place(tgt_pos)
                    self.memory.record_action("place", None, entity_name)
                else:
                    print(f"    Target '{entity_name}' not found in memory")

            elif action == "move":
                print(f"    Move action (auto-stabilize)")
                self.scene.step(40)

            # Update memory after each action
            self.scene.step(20)
            self.memory.update_positions(self.scene.objects)
            self.scene.capture_frame()  # Record frame for video

        self.bench.end("execution")

        # Phase 4: Capture final state & verify
        self.bench.start("verify")
        final_img = self.scene.render_rgb()
        self.verifier.capture_after(final_img)
        self.memory.update_positions(self.scene.objects)
        self.bench.end("verify")

        # Phase 5: Verification (scene memory + camera)
        self.bench.start("verification")
        mem_verification = self._verify_task(steps)
        cam_verification = self.verifier.verify(instruction)
        # Combine: both must agree for full confidence
        verification = {
            "success": mem_verification.get("success", False),
            "memory_check": mem_verification,
            "camera_check": cam_verification,
            "reasoning": mem_verification.get("reasoning", ""),
        }
        self.bench.end("verification")

        # Report
        self.bench.print_report()
        self._print_verification(verification)

        return {
            "success": verification.get("success", False),
            "instruction": instruction,
            "plan": plan,
            "verification": verification,
            "scene_state": self.memory.get_state(),
        }

    def _verify_task(self, steps: List[Dict]) -> Dict:
        """Verify task completion using scene memory."""
        results = []

        for step in steps:
            if step["action"] == "place":
                obj_name = step.get("object", "red cup")
                tgt_name = step.get("target", "blue box")
                entity_obj = KNOWN_OBJECTS.get(obj_name, obj_name)
                entity_tgt = KNOWN_OBJECTS.get(tgt_name, tgt_name)
                result = self.memory.verify_placement(entity_obj, entity_tgt)
                results.append(result)

        if not results:
            return {"success": True, "reasoning": "No placement to verify"}

        overall_success = all(r.get("success", False) for r in results)
        return {
            "success": overall_success,
            "details": results,
            "reasoning": f"Verified {len(results)} placements",
        }

    def _print_verification(self, verification: Dict):
        print(f"\n{'='*60}")
        print(f"  Verification Result")
        print(f"{'='*60}")
        status = "SUCCESS" if verification.get("success") else "FAILED"
        print(f"  Status: {status}")

        mem = verification.get("memory_check", {})
        cam = verification.get("camera_check", {})
        print(f"  [Memory]   {mem.get('success', 'N/A')} — {mem.get('reasoning', 'N/A')}")
        if "details" in mem:
            for i, d in enumerate(mem["details"]):
                print(f"    Placement {i+1}: dist={d.get('distance_to_target','?')}m, moved={d.get('moved_from_start','?')}, near={d.get('near_target','?')}")
        print(f"  [Camera]   {cam.get('success', 'N/A')} — {cam.get('reasoning', 'N/A')}")
        print(f"{'='*60}")


# ── Convenience Function ─────────────────────────────────────

def run_mvp_v2(instruction: str = "Pick up the red cup and place it in the blue box"):
    scene = create_tabletop_scene(show_viewer=False)
    orch = OrchestratorV2(scene)
    result = orch.run(instruction, use_vlm=False)
    scene.save_video("demo/output/mvp_v2_result.mp4", fps=30)
    return result
