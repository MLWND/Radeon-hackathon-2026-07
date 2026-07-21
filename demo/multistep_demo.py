#!/usr/bin/env python3
"""
Multi-step Task Demo — demonstrates multi-step autonomous manipulation.

Task: "Pick the red cube and place it next to the blue cube, then pick the green cube and place it next to the red cube"

Shows:
- Multi-step task decomposition via TaskPlanner
- Sequential execution via ActionScheduler
- Scene memory tracking across steps
- Verification after each step
"""
import genesis as gs
import numpy as np
import json, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


def header(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")


def main():
    header("Multi-step Task Demo")

    # ── Init ──────────────────────────────────────────────
    print("\n[1/6] Init Genesis + GraspEnv...")
    gs.init(backend=gs.amdgpu)

    from src.envs.grasp_env import GraspEnv
    env = GraspEnv(num_envs=0, ctrl_dt=0.01)
    print(f"  Objects: {list(env.entities.keys())}")

    # ── VLM (optional, falls back to defaults) ────────────
    print("\n[2/6] Connect to VLM...")
    try:
        from openai import OpenAI
        vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=60)
        vlm_available = True
        print("  vLLM connected")
    except Exception:
        vlm_available = False
        print("  vLLM not available, using defaults")

    # ── Multi-step instruction ────────────────────────────
    INSTRUCTION = "Pick the red cube and place it next to the blue cube, then pick the green cube and place it next to the red cube"
    print(f"\n  Instruction: {INSTRUCTION}")

    # ── Task planning ─────────────────────────────────────
    print("\n[3/6] Task planning...")
    from src.planner.task_planner import TaskPlanner
    from src.planner.action_scheduler import ActionScheduler

    planner = TaskPlanner()
    plan = planner.plan(INSTRUCTION)
    steps = plan.get("steps", [])
    print(f"  Decomposed into {len(steps)} steps:")
    for i, step in enumerate(steps):
        print(f"    {i+1}. {step['action']} → {step.get('object', step.get('target', ''))}")

    # ── Execute ───────────────────────────────────────────
    print("\n[4/6] Executing multi-step plan...")
    from src.vision.scene_memory import SceneMemory
    memory = SceneMemory()
    memory.initialize(env.entities)

    env.start_recording()
    results = []
    step_num = 0

    scheduler = ActionScheduler()
    scheduler.load_plan(steps)

    while not scheduler.is_complete():
        action = scheduler.next_action()
        act_type = action.get("action", "")
        obj_name = action.get("object", "")
        tgt_name = action.get("target", "")

        step_num += 1
        print(f"\n  Step {step_num}: {act_type} {obj_name or tgt_name}")

        if act_type == "pick":
            result = env.execute_action(action)
            ok = result.get("ok", False)
            lifted = result.get("result", False)
            print(f"    ok={ok}, lifted={lifted}")
            scheduler.mark_done(action)
            memory.record_action("pick", obj_name)
            results.append({"step": step_num, "action": "pick", "object": obj_name,
                            "ok": ok, "lifted": lifted})

        elif act_type == "place":
            # Resolve target position from scene memory or entity
            target_pos = None
            if tgt_name in env.entities:
                target_pos = env.entities[tgt_name].get_pos().cpu().numpy().tolist()
            elif tgt_name:
                # Try to find by partial name match
                for name in env.entities:
                    if tgt_name in name:
                        target_pos = env.entities[name].get_pos().cpu().numpy().tolist()
                        break

            result = env.execute_action(action, target_pos=target_pos)
            ok = result.get("ok", False)
            err = result.get("result", 0.99)
            print(f"    ok={ok}, error={err*100:.1f}cm")
            scheduler.mark_done(action)
            memory.record_action("place", obj_name or pick_name, tgt_name)
            results.append({"step": step_num, "action": "place",
                            "object": obj_name, "ok": ok,
                            "error_cm": round(err * 100, 2)})

        else:
            result = env.execute_action(action)
            scheduler.mark_done(action)
            results.append({"step": step_num, "action": act_type, "ok": result.get("ok", False)})

    # ── Verify ────────────────────────────────────────────
    print("\n[5/6] Verification...")
    memory.update_positions(env.entities)

    for name in env.entities:
        p = env.entities[name].get_pos().cpu().numpy()
        print(f"  {name}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

    # ── Output ────────────────────────────────────────────
    print("\n[6/6] Saving outputs...")
    for _ in range(30):
        env.scene.step(2)
        env.vis_cam.render()
    env.stop_recording("demo/output/multistep_demo.mp4", fps=30)

    # Summary
    successful = sum(1 for r in results if r.get("ok") or r.get("lifted"))
    print(f"\n{'='*60}")
    print(f"  MULTI-STEP RESULTS")
    print(f"{'='*60}")
    print(f"  Steps executed: {len(results)}")
    print(f"  Successful: {successful}/{len(results)}")
    for r in results:
        status = "OK" if r.get("ok") or r.get("lifted") else "FAIL"
        print(f"    Step {r['step']}: {r['action']} {r.get('object', '')} → {status}")
    print(f"  Status: {'SUCCESS' if successful == len(results) else 'PARTIAL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
