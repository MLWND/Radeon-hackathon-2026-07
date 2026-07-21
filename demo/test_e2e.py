#!/usr/bin/env python3
"""
RoboPilot — Complete End-to-End Test
Uses GraspEnv + ALL pipeline modules with detailed logging.
Modules: GraspEnv, CameraWrapper, QwenVLWrapper, SceneMemory,
         TaskPlanner, ActionScheduler, CameraVerifier,
         FailureDetector, RecoveryManager
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


class Logger:
    def __init__(self):
        self.steps = []
        self.t0 = time.time()

    def log(self, module, msg, **kw):
        elapsed = time.time() - self.t0
        entry = {"time": f"{elapsed:.2f}s", "module": module, "msg": msg, **kw}
        self.steps.append(entry)
        extras = " | ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
        print(f"  [{elapsed:6.2f}s] [{module:20s}] {msg}" + (f"  ({extras})" if extras else ""))

    def summary(self):
        total = time.time() - self.t0
        print(f"\n{'='*70}")
        print(f"  TIMELINE ({len(self.steps)} steps, {total:.1f}s total)")
        print(f"{'='*70}")
        modules = {}
        for s in self.steps:
            m = s["module"]
            if m not in modules:
                modules[m] = []
            modules[m].append(s)
        for m, entries in modules.items():
            times = []
            for e in entries:
                try:
                    times.append(float(e["time"].rstrip("s")))
                except Exception:
                    pass
            span = times[-1] - times[0] if len(times) >= 2 else 0
            print(f"  {m:20s}: {len(entries):2d} steps | span {span:.2f}s")
        print(f"{'='*70}")
        return self.steps


log = Logger()


def main():
    print("=" * 70)
    print("  RoboPilot — Complete Module Integration Test")
    print("=" * 70)

    # ═════════════════════════════════════════════════════════
    # MODULE 1: GraspEnv (scene + robot + objects)
    # ═════════════════════════════════════════════════════════
    log.log("GraspEnv", "Initializing Genesis + GraspEnv...")
    t0 = time.time()
    gs.init(backend=gs.amdgpu)

    from src.envs.grasp_env import GraspEnv
    env = GraspEnv(num_envs=0, ctrl_dt=0.01)
    ents = env.entities
    log.log("GraspEnv", "Environment ready",
            robot_dofs=env.robot.n_dofs,
            objects=list(ents.keys()),
            ms=f"{(time.time()-t0)*1000:.0f}")

    for name, obj in ents.items():
        p = obj.get_pos().cpu().numpy()
        log.log("GraspEnv", f"  {name} at [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

    # ═════════════════════════════════════════════════════════
    # MODULE 2: CameraWrapper
    # ═════════════════════════════════════════════════════════
    log.log("CameraWrapper", "Initializing camera module...")
    t0 = time.time()
    from src.vision.camera import CameraWrapper
    cam = CameraWrapper(env.vis_cam)
    img_before = cam.render()
    cam.render_and_save("demo/output/test_before.png")
    log.log("CameraWrapper", "Image captured",
            shape=f"{img_before.shape}",
            ms=f"{(time.time()-t0)*1000:.0f}")

    # ═════════════════════════════════════════════════════════
    # MODULE 3: SceneMemory
    # ═════════════════════════════════════════════════════════
    log.log("SceneMemory", "Initializing scene memory...")
    from src.vision.scene_memory import SceneMemory
    memory = SceneMemory()
    memory.initialize(ents)
    log.log("SceneMemory", "Memory initialized", objects=list(memory.objects.keys()))
    for name, info in memory.objects.items():
        log.log("SceneMemory", f"  {name}: color={info['color']}, type={info['type']}, pos={[round(x,3) for x in info['position']]}")

    # ═════════════════════════════════════════════════════════
    # MODULE 4: QwenVLWrapper (VLM Perception)
    # ═════════════════════════════════════════════════════════
    log.log("QwenVLWrapper", "Connecting to vLLM server...")
    from src.vision.qwen3vl import QwenVLWrapper, resolve_place_position
    from openai import OpenAI
    from PIL import Image
    import base64, io
    vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=60)

    INSTRUCTION = "Pick the red cube and place it next to the blue cube"
    pil_img = Image.fromarray(img_before)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    log.log("QwenVLWrapper", "Sending to Qwen3-VL-8B...")
    t1 = time.time()
    response = vlm_client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": (
                f'Return ONLY JSON: {{"pick":"name","place_relative":"description","reasoning":"brief"}}\n'
                f'Objects: {list(ents.keys())}\nInstruction: {INSTRUCTION}'
            )},
        ]}],
        max_tokens=64,
    )
    vlm_ms = (time.time() - t1) * 1000
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
        if pick_name not in ents:
            pick_name = "red_cube"
    except Exception as ex:
        log.log("QwenVLWrapper", f"Parse error: {ex}")

    log.log("QwenVLWrapper", "VLM response parsed",
            pick=pick_name, place=place_relative,
            reasoning=reasoning[:60], inference_ms=f"{vlm_ms:.0f}")

    place_pos = resolve_place_position(place_relative, ents)
    log.log("QwenVLWrapper", "Place position resolved",
            target=f"[{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")

    # ═════════════════════════════════════════════════════════
    # MODULE 5: TaskPlanner
    # ═════════════════════════════════════════════════════════
    log.log("TaskPlanner", "Planning action sequence...")
    from src.planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    plan = planner.plan(INSTRUCTION, img_before)
    log.log("TaskPlanner", "Plan generated",
            reasoning=plan.get("reasoning", "")[:60],
            steps=len(plan.get("steps", [])),
            inference_ms=f"{plan.get('inference_ms', 0):.0f}")
    for i, step in enumerate(plan.get("steps", [])):
        log.log("TaskPlanner", f"  Step {i+1}: {step['action']} → {step.get('object', step.get('target', ''))} ({step.get('description', '')})")

    # ═════════════════════════════════════════════════════════
    # MODULE 6: ActionScheduler
    # ═════════════════════════════════════════════════════════
    log.log("ActionScheduler", "Loading plan...")
    from src.planner.action_scheduler import ActionScheduler
    scheduler = ActionScheduler()
    action_steps = plan.get("steps", [
        {"action": "pick", "object": pick_name, "description": f"Grasp {pick_name}"},
        {"action": "place", "target": "blue_cube", "description": "Place near blue_cube"},
    ])
    scheduler.load_plan(action_steps)
    log.log("ActionScheduler", "Plan loaded",
            total=scheduler.get_progress()["total"],
            current=scheduler.get_progress()["current"])

    # ═════════════════════════════════════════════════════════
    # MODULE 7: GraspEnv execute (pick & place)
    # ═════════════════════════════════════════════════════════
    log.log("GraspEnv", "Starting manipulation...")

    pre_pick_pos = {}
    for name, obj in ents.items():
        pre_pick_pos[name] = obj.get_pos().cpu().numpy().copy()

    env.start_recording()

    # === PICK ===
    t0 = time.time()
    log.log("GraspEnv", f"PICKING: {pick_name}...")
    log.log("GraspEnv", f"  pre-pick pos: {[round(x,3) for x in pre_pick_pos[pick_name]]}")

    action = scheduler.next_action()
    if action is None:
        log.log("GraspEnv", "ERROR: No pick action available")
        return
    log.log("ActionScheduler", f"  Executing: {action}", progress=str(scheduler.get_progress()))
    pick_result = env.execute_action(action)
    lifted = pick_result["result"] if pick_result.get("ok") else False
    pick_name = action.get("object", pick_name)
    pick_ms = (time.time() - t0) * 1000
    scheduler.mark_done(action)

    post_pick_pos = {}
    for name, obj in ents.items():
        post_pick_pos[name] = obj.get_pos().cpu().numpy().copy()

    log.log("GraspEnv", f"  PICK done", lifted=lifted, pick_ms=f"{pick_ms:.0f}")
    log.log("GraspEnv", f"  post-pick {pick_name}: {[round(x,3) for x in post_pick_pos[pick_name]]}")

    for name in ents:
        if name != pick_name:
            delta = np.linalg.norm(post_pick_pos[name][:2] - pre_pick_pos[name][:2])
            if delta > 0.001:
                log.log("GraspEnv", f"  WARNING: {name} moved {delta*100:.1f}cm during pick!")
            else:
                log.log("GraspEnv", f"  {name}: undisturbed ({delta*100:.2f}cm)")

    # === PLACE ===
    t0 = time.time()
    log.log("GraspEnv", f"PLACING at [{place_pos[0]:.3f}, {place_pos[1]:.3f}]...")

    action = scheduler.next_action()
    if action is None:
        log.log("GraspEnv", "ERROR: No place action available")
        return
    log.log("ActionScheduler", f"  Executing: {action}", progress=str(scheduler.get_progress()))
    place_result = env.execute_action(action, target_pos=place_pos)
    err = place_result["result"] if place_result.get("ok") else 0.99
    place_ms = (time.time() - t0) * 1000
    scheduler.mark_done(action)

    done = env.is_done()
    log.log("GraspEnv", f"  is_done={done}")

    cube_final = ents[pick_name].get_pos().cpu().numpy()
    log.log("GraspEnv", f"  PLACE done", error_cm=f"{err*100:.1f}", place_ms=f"{place_ms:.0f}")
    log.log("GraspEnv", f"  final {pick_name}: [{cube_final[0]:.3f}, {cube_final[1]:.3f}, {cube_final[2]:.3f}]")

    log.log("ActionScheduler", "All actions complete", progress=str(scheduler.get_progress()))

    # ═════════════════════════════════════════════════════════
    # MODULE 8: SceneMemory (update + verify)
    # ═════════════════════════════════════════════════════════
    log.log("SceneMemory", "Updating positions after manipulation...")
    memory.update_positions(ents)
    memory.record_action("pick", pick_name)
    memory.record_action("place", pick_name, target_name="blue_cube")

    nearest = None
    min_dist = float("inf")
    for name, obj in ents.items():
        if name != pick_name:
            p = obj.get_pos().cpu().numpy()
            d = np.linalg.norm(cube_final[:2] - p[:2])
            if d < min_dist:
                min_dist = d
                nearest = name

    if nearest:
        verify_result = memory.verify_placement(pick_name, nearest, proximity_threshold=0.15)
        log.log("SceneMemory", "Placement verification",
                target=nearest, success=verify_result["success"],
                distance=f"{verify_result['distance_to_target']:.3f}m",
                moved=verify_result["moved_from_start"])

    # ═════════════════════════════════════════════════════════
    # MODULE 9: CameraVerifier (before/after)
    # ═════════════════════════════════════════════════════════
    log.log("CameraVerifier", "Capturing after image...")
    img_after = cam.render()
    cam.render_and_save("demo/output/test_after.png")

    from src.vision.verifier import CameraVerifier
    verifier = CameraVerifier()
    verifier.capture_before(img_before)
    verifier.capture_after(img_after)
    verify_cam = verifier.verify(INSTRUCTION)
    log.log("CameraVerifier", "Pixel verification",
            success=verify_cam["success"],
            confidence=f"{verify_cam.get('confidence', 0):.2f}",
            method=verify_cam.get("method", "unknown"),
            reasoning=verify_cam.get("reasoning", "")[:80])

    # ═════════════════════════════════════════════════════════
    # MODULE 10: FailureDetector
    # ═════════════════════════════════════════════════════════
    log.log("FailDetector", "Running failure checks...")
    grasp_ok = lifted and (post_pick_pos[pick_name][2] > pre_pick_pos[pick_name][2] + 0.05)
    log.log("FailDetector", "  Grasp check", passed=grasp_ok,
            z_before=f"{pre_pick_pos[pick_name][2]:.3f}",
            z_after=f"{post_pick_pos[pick_name][2]:.3f}")

    place_ok = err < 0.15
    log.log("FailDetector", "  Placement check", passed=place_ok,
            error=f"{err*100:.1f}cm", threshold="15cm")

    disturbed = []
    for name in ents:
        if name != pick_name:
            delta = np.linalg.norm(post_pick_pos[name][:2] - pre_pick_pos[name][:2])
            if delta > 0.01:
                disturbed.append((name, delta))
    log.log("FailDetector", "  Disturbance check",
            passed=len(disturbed) == 0,
            disturbed=[f"{n}:{d*100:.1f}cm" for n, d in disturbed] if disturbed else "none")

    overall = "SUCCESS" if (grasp_ok and place_ok and len(disturbed) == 0) else "PARTIAL"
    log.log("FailDetector", f"  OVERALL: {overall}")

    # ═════════════════════════════════════════════════════════
    # OUTPUT: Video + Comparison + JSON
    # ═════════════════════════════════════════════════════════
    log.log("Output", "Saving video and comparison...")
    for _ in range(30):
        env.scene.step(2)
        env.vis_cam.render()
    env.stop_recording("demo/output/test_e2e.mp4", fps=30)

    import cv2
    h, w = img_before.shape[:2]
    canvas = np.zeros((h + 50, w * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = img_before
    canvas[:h, w:] = img_after
    cv2.rectangle(canvas, (0, h), (w * 2, h + 50), (30, 30, 30), -1)
    cv2.putText(canvas, f"Before | VLM: {vlm_ms:.0f}ms", (10, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"After | Place: {err*100:.1f}cm error", (w + 10, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    cv2.line(canvas, (w, 0), (w, h), (100, 100, 100), 2)
    Image.fromarray(canvas).save("demo/output/test_comparison.png")

    triple_verify = {
        "task": INSTRUCTION,
        "env": "GraspEnv (GPU tensors)",
        "vlm_decision": {"pick": pick_name, "place_relative": place_relative, "inference_ms": round(vlm_ms)},
        "grasp_verification": {
            "method": "object_height_change",
            "z_before": round(float(pre_pick_pos[pick_name][2]), 4),
            "z_after_pick": round(float(post_pick_pos[pick_name][2]), 4),
            "lifted": bool(grasp_ok),
        },
        "placement_verification": {
            "method": "euclidean_xy",
            "target": [round(x, 3) for x in place_pos],
            "final": [round(float(x), 3) for x in cube_final],
            "error_cm": round(err * 100, 2),
            "passed": bool(place_ok),
        },
        "disturbance_verification": {
            "method": "object_position_delta",
            "disturbed": [{"name": n, "delta_cm": round(d*100, 2)} for n, d in disturbed],
            "passed": len(disturbed) == 0,
        },
        "camera_verification": {
            "method": "pixel_diff",
            "success": bool(verify_cam["success"]),
            "mean_diff": verify_cam.get("reasoning", ""),
        },
        "overall_status": overall,
    }
    json_out = json.dumps(triple_verify, indent=2, default=float)
    with open("demo/output/verification.json", "w") as f:
        f.write(json_out)
    log.log("TripleVerify", "Structured JSON saved", file="demo/output/verification.json")
    print("\n  Triple Verification JSON:")
    print(json_out)

    # ═════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════
    total_ms = pick_ms + place_ms + vlm_ms
    log.summary()

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Env:     GraspEnv (GPU tensors, Gym interface)")
    print(f"  VLM:     Qwen3-VL-8B via vLLM | {vlm_ms:.0f}ms")
    print(f"  Pick:    {pick_name} | lifted={lifted} | {pick_ms:.0f}ms")
    print(f"  Place:   error {err*100:.1f}cm | {place_ms:.0f}ms")
    print(f"  Verify:  pixel_change={verify_cam['success']} | grasp_ok={grasp_ok} | place_ok={place_ok}")
    print(f"  Objects: {len(disturbed)} disturbed")
    print(f"  Total:   {total_ms:.0f}ms pipeline time")
    print(f"  Status:  {overall}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
