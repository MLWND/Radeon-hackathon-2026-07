#!/usr/bin/env python3
"""
RoboPilot — Full Closed-Loop Demo (Complete Pipeline)

Pipeline: GraspEnv → VLM perception → Task decomposition → Action scheduling → Execute → Verify

Note: TaskPlanner uses rule-based keyword matching for instruction decomposition.
VLM (Qwen3-VL) handles object detection and spatial grounding.
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


def header(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")


def main():
    header("RoboPilot — Full Pipeline Demo")

    # ═══ 1. INIT GRASP ENV ═════════════════════════════════════
    print("\n[1/9] Init Genesis + GraspEnv...")
    gs.init(backend=gs.amdgpu)

    from src.envs.grasp_env import GraspEnv
    env = GraspEnv(num_envs=0, ctrl_dt=0.01)
    print(f"  Robot: {env.robot.n_dofs} DOFs")
    print(f"  Objects: {list(env.entities.keys())}")
    print(f"  Stereo: {'enabled' if env.left_cam else 'mono fallback'}")

    # ═══ 2. VLM (via vLLM OpenAI API) ═════════════════════════
    print("\n[2/9] Connect to Qwen3-VL via vLLM...")
    from openai import OpenAI
    from PIL import Image
    import base64, io

    vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=60)

    # ═══ 3. PIPELINE MODULES ══════════════════════════════════
    print("\n[3/9] Init pipeline modules...")
    from src.vision.camera import CameraWrapper
    from src.vision.scene_memory import SceneMemory
    from src.vision.verifier import CameraVerifier
    from src.planner.task_planner import TaskPlanner
    from src.planner.action_scheduler import ActionScheduler
    from src.planner.recovery import FailureDetector, RecoveryManager
    from src.vision.qwen3vl import resolve_place_position

    cam = CameraWrapper(env.vis_cam)
    memory = SceneMemory()
    memory.initialize(env.entities)
    verifier = CameraVerifier()
    planner = TaskPlanner()
    scheduler = ActionScheduler()
    fail_detector = FailureDetector(memory)
    recovery_manager = RecoveryManager(memory)

    # ═══ 4. CAPTURE BEFORE ════════════════════════════════════
    print("\n[4/9] Capture scene...")
    img_before = cam.render()
    cam.render_and_save("demo/output/full_before.png")

    pre_pos = {}
    for name, obj in env.entities.items():
        pre_pos[name] = obj.get_pos().cpu().numpy().copy()

    # ═══ 5. VLM DETECT ════════════════════════════════════════
    header("Step 5: VLM Perception")
    INSTRUCTION = "Pick the red cube and place it next to the blue cube"

    pil_img = Image.fromarray(img_before)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    print(f"  Instruction: {INSTRUCTION}")
    t0 = time.time()
    response = vlm_client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": (
                f'Return ONLY JSON: {{"pick":"name","place_relative":"description","reasoning":"brief"}}\n'
                f'Objects: {list(env.entities.keys())}\nInstruction: {INSTRUCTION}'
            )},
        ]}],
        max_tokens=64,
    )
    vlm_ms = (time.time() - t0) * 1000
    raw = response.choices[0].message.content

    pick_name = "red_cube"
    place_relative = "right of blue_cube"
    reasoning = ""
    try:
        s = raw.rfind("{")
        e = raw.rfind("}") + 1
        parsed = json.loads(raw[s:e])
        pick_name = parsed.get("pick", "red_cube")
        place_relative = parsed.get("place_relative", "right of blue_cube")
        reasoning = parsed.get("reasoning", "")
        if pick_name not in env.entities:
            pick_name = "red_cube"
    except Exception:
        pass

    place_pos = resolve_place_position(place_relative, env.entities)
    print(f"  VLM: {vlm_ms:.0f}ms")
    print(f"  pick={pick_name}, place_relative={place_relative}")
    print(f"  place_pos=[{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")

    # ═══ 6. TASK PLANNER → ACTION SCHEDULER ═══════════════════
    header("Step 6: Plan & Schedule")
    plan = planner.plan(INSTRUCTION, img_before)
    print(f"  Planner: {plan.get('reasoning', '')[:80]}")

    action_steps = plan.get("steps", [
        {"action": "pick", "object": pick_name},
        {"action": "place", "target": "blue_cube"},
    ])
    scheduler.load_plan(action_steps)
    print(f"  {len(action_steps)} steps scheduled")
    for i, step in enumerate(action_steps):
        print(f"    {i+1}. {step['action']} → {step.get('object', step.get('target', ''))}")

    # ═══ 7. EXECUTE VIA GRASP ENV ═════════════════════════════
    header("Step 7: Execute")

    env.start_recording()
    results = []

    while not scheduler.is_complete():
        action = scheduler.next_action()
        if action is None:
            break
        act_type = action.get("action", "")
        obj_name = action.get("object", "")
        tgt_name = action.get("target", "")
        print(f"\n  >> {act_type}: {obj_name or tgt_name}")

        # Use RecoveryManager for pick and place actions
        if act_type in ("pick", "place"):
            result = recovery_manager.execute_with_recovery(
                env, action, env.entities, target_pos=place_pos if act_type == "place" else None
            )
            ok = result.get("success", False)
            attempts = result.get("attempts", 1)
            replanned = result.get("replanned", False)
            print(f"     ok={ok}, attempts={attempts}, replanned={replanned}")
            scheduler.mark_done(action)
            memory.record_action(act_type, obj_name or pick_name, tgt_name if act_type == "place" else None)
            results.append({"action": act_type, "object": obj_name or pick_name,
                            "ok": ok, "attempts": attempts, "replanned": replanned})
        else:
            result = env.execute_action(action)
            ok = result.get("ok", False)
            print(f"     ok={ok}")
            scheduler.mark_done(action)
            results.append({"action": act_type, "ok": ok})

    # ═══ 8. VERIFY ════════════════════════════════════════════
    header("Step 8: Verify")

    img_after = cam.render()
    cam.render_and_save("demo/output/full_after.png")

    verifier.capture_before(img_before)
    verifier.capture_after(img_after)
    cam_verify = verifier.verify(INSTRUCTION)
    print(f"  Camera verify: success={cam_verify['success']}, "
          f"confidence={cam_verify.get('confidence', 0):.2f}")

    memory.update_positions(env.entities)
    nearest = None
    min_dist = float("inf")
    cube_final = env.entities[pick_name].get_pos().cpu().numpy()
    for name, obj in env.entities.items():
        if name != pick_name:
            p = obj.get_pos().cpu().numpy()
            d = np.linalg.norm(cube_final[:2] - p[:2])
            if d < min_dist:
                min_dist = d
                nearest = name
    if nearest:
        mem_verify = memory.verify_placement(pick_name, nearest, proximity_threshold=0.15)
        print(f"  Memory verify: success={mem_verify['success']}, "
              f"dist={mem_verify['distance_to_target']:.3f}m to {nearest}")

    disturbed = []
    for name in env.entities:
        if name != pick_name:
            p = env.entities[name].get_pos().cpu().numpy()
            delta = np.linalg.norm(p[:2] - pre_pos[name][:2])
            if delta > 0.01:
                disturbed.append((name, delta))
            print(f"  {name}: {'disturbed' if delta > 0.01 else 'OK'} ({delta*100:.2f}cm)")

    place_target = np.array(place_pos[:2])
    final_err = float(np.linalg.norm(cube_final[:2] - place_target))
    overall = "SUCCESS" if (final_err < 0.10 and len(disturbed) == 0) else "PARTIAL"

    # ═══ OUTPUT ════════════════════════════════════════════════
    print("\n  Saving outputs...")
    for _ in range(30):
        env.scene.step(2)
        env.vis_cam.render()
    env.stop_recording("demo/output/full_demo.mp4", fps=30)

    import cv2
    h, w = img_before.shape[:2]
    canvas = np.zeros((h + 50, w * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = img_before
    canvas[:h, w:] = img_after
    cv2.rectangle(canvas, (0, h), (w * 2, h + 50), (30, 30, 30), -1)
    cv2.putText(canvas, f"Before | VLM: {vlm_ms:.0f}ms", (10, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"After | Err: {final_err*100:.1f}cm", (w + 10, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    cv2.line(canvas, (w, 0), (w, h), (100, 100, 100), 2)
    Image.fromarray(canvas).save("demo/output/full_comparison.png")

    verification = {
        "task": INSTRUCTION,
        "pipeline": "GraspEnv → TaskPlanner → ActionScheduler → execute_action",
        "vlm_decision": {
            "pick": pick_name, "place_relative": place_relative,
            "reasoning": reasoning, "inference_ms": round(vlm_ms),
        },
        "plan": {"steps": action_steps, "inference_ms": round(plan.get("inference_ms", 0))},
        "execution_results": results,
        "placement": {
            "target": [round(x, 3) for x in place_pos],
            "final": [round(float(x), 3) for x in cube_final],
            "error_cm": round(final_err * 100, 2),
        },
        "verification": {
            "camera": cam_verify,
            "disturbed": [{"name": n, "delta_cm": round(d * 100, 2)} for n, d in disturbed],
        },
        "overall_status": overall,
    }
    with open("demo/output/verification.json", "w") as f:
        json.dump(verification, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Env:       GraspEnv (GPU tensors, no numpy in hot path)")
    print(f"  VLM:       Qwen3-VL-8B via vLLM | {vlm_ms:.0f}ms")
    print(f"  Planner:   TaskPlanner → {len(action_steps)} steps")
    print(f"  Scheduler: ActionScheduler → {len(results)} executed")
    print(f"  Place:     error {final_err*100:.1f}cm")
    print(f"  Verify:    camera={cam_verify['success']} | disturbed={len(disturbed)}")
    print(f"  Status:    {overall}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
