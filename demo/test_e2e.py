#!/usr/bin/env python3
"""
RoboPilot — Complete End-to-End Test
Uses ALL existing modules with detailed logging.
Modules: SceneManager, CameraWrapper, QwenVLWrapper, SceneMemory,
         TaskPlanner, ActionScheduler, ManipulationPipeline,
         CameraVerifier, FailureDetector + RecoveryManager
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)

# ── Logging ────────────────────────────────────────────────
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
                except:
                    pass
            if len(times) >= 2:
                span = times[-1] - times[0]
            else:
                span = 0
            print(f"  {m:20s}: {len(entries):2d} steps | span {span:.2f}s")
        print(f"{'='*70}")
        return self.steps

log = Logger()


def main():
    print("=" * 70)
    print("  RoboPilot — Complete Module Integration Test")
    print("=" * 70)

    # ═════════════════════════════════════════════════════════
    # MODULE 1: SceneManager (Genesis Scene)
    # ═════════════════════════════════════════════════════════
    log.log("SceneManager", "Initializing Genesis...")
    t0 = time.time()
    gs.init(backend=gs.amdgpu)
    log.log("SceneManager", "Genesis init done", ms=f"{(time.time()-t0)*1000:.0f}")

    t0 = time.time()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True, constraint_solver=gs.constraint_solver.Newton),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=os.path.join(
        "venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml")))

    ents = {}
    ents["red_cube"] = scene.add_entity(gs.morphs.Box(size=(0.04,0.04,0.04), pos=(0.65, 0.0, 0.02)),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)))
    ents["blue_cube"] = scene.add_entity(gs.morphs.Box(size=(0.04,0.04,0.04), pos=(0.4, 0.2, 0.02)),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)))
    ents["green_cube"] = scene.add_entity(gs.morphs.Box(size=(0.04,0.04,0.04), pos=(0.7, -0.1, 0.02)),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)))

    vis_cam = scene.add_camera(res=(1280, 720), pos=(1.5, -2.0, 1.6), lookat=(0.5, 0, 0.0), fov=45)
    scene.build()
    scene.step(200)
    log.log("SceneManager", "Scene built",
            entities=len(ents),
            robot_dofs=robot.n_dofs,
            build_ms=f"{(time.time()-t0)*1000:.0f}")

    # Print initial object positions
    for name, obj in ents.items():
        p = obj.get_pos().cpu().numpy()
        log.log("SceneManager", f"  {name} at [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

    # ═════════════════════════════════════════════════════════
    # MODULE 2: CameraWrapper
    # ═════════════════════════════════════════════════════════
    log.log("CameraWrapper", "Initializing camera module...")
    t0 = time.time()
    from src.vision.camera import CameraWrapper
    cam = CameraWrapper(vis_cam)
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
    t0 = time.time()
    from src.vision.qwen3vl import QwenVLWrapper, resolve_place_position
    vlm = QwenVLWrapper()

    # vLLM mode (no local model load, uses OpenAI API)
    from openai import OpenAI
    from PIL import Image
    import base64, io
    vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=60)

    INSTRUCTION = "Pick the red cube and place it next to the blue cube"
    pil_img = Image.fromarray(img_before)
    buf = io.BytesIO(); pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    log.log("QwenVLWrapper", "Sending to Qwen3-VL-8B...")
    t1 = time.time()
    response = vlm_client.chat.completions.create(model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},
            {"type":"text","text":f'Return ONLY JSON: {{"pick":"name","place_relative":"description","reasoning":"brief"}}\nObjects: {list(ents.keys())}\nInstruction: {INSTRUCTION}'}
        ]}], max_tokens=64)
    vlm_ms = (time.time() - t1) * 1000
    raw = response.choices[0].message.content

    # Parse VLM output
    pick_name = "red_cube"
    place_relative = "right of blue_cube"
    reasoning = ""
    try:
        s = raw.rfind("{"); e = raw.rfind("}") + 1
        parsed = json.loads(raw[s:e])
        pick_name = parsed.get("pick", "red_cube")
        place_relative = parsed.get("place_relative", "right of blue_cube")
        reasoning = parsed.get("reasoning", "")
        if pick_name not in ents:
            pick_name = "red_cube"
    except Exception as ex:
        log.log("QwenVLWrapper", f"Parse error: {ex}")

    log.log("QwenVLWrapper", "VLM response parsed",
            pick=pick_name,
            place=place_relative,
            reasoning=reasoning[:60],
            inference_ms=f"{vlm_ms:.0f}")

    # Resolve place position via scene memory
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
    # MODULE 7: ManipulationPipeline (pick & place)
    # ═════════════════════════════════════════════════════════
    log.log("ManipPipeline", "Initializing manipulation pipeline...")
    from src.control.primitives import ManipulationPipeline
    pipe = ManipulationPipeline(robot, scene, ents)

    # Record pre-pick positions
    pre_pick_pos = {}
    for name, obj in ents.items():
        pre_pick_pos[name] = obj.get_pos().cpu().numpy().copy()

    vis_cam.start_recording()

    # === PICK ===
    t0 = time.time()
    log.log("ManipPipeline", f"PICKING: {pick_name}...")
    log.log("ManipPipeline", f"  pre-pick pos: {[round(x,3) for x in pre_pick_pos[pick_name]]}")

    action = scheduler.next_action()  # pick action
    log.log("ActionScheduler", f"  Executing: {action}", progress=str(scheduler.get_progress()))

    lifted = pipe.suction_pick(pick_name)
    pick_ms = (time.time() - t0) * 1000
    scheduler.mark_done(action)

    # Record post-pick positions
    post_pick_pos = {}
    for name, obj in ents.items():
        post_pick_pos[name] = obj.get_pos().cpu().numpy().copy()

    log.log("ManipPipeline", f"  PICK done",
            lifted=lifted,
            pick_ms=f"{pick_ms:.0f}")
    log.log("ManipPipeline", f"  post-pick {pick_name}: {[round(x,3) for x in post_pick_pos[pick_name]]}")

    # Check other objects moved
    for name in ents:
        if name != pick_name:
            delta = np.linalg.norm(post_pick_pos[name][:2] - pre_pick_pos[name][:2])
            if delta > 0.001:
                log.log("ManipPipeline", f"  WARNING: {name} moved {delta*100:.1f}cm during pick!")
            else:
                log.log("ManipPipeline", f"  {name}: undisturbed ({delta*100:.2f}cm)")

    # === PLACE ===
    t0 = time.time()
    log.log("ManipPipeline", f"PLACING at [{place_pos[0]:.3f}, {place_pos[1]:.3f}]...")

    action = scheduler.next_action()  # place action
    log.log("ActionScheduler", f"  Executing: {action}", progress=str(scheduler.get_progress()))

    err = pipe.suction_place(pick_name, place_pos)
    place_ms = (time.time() - t0) * 1000
    scheduler.mark_done(action)

    cube_final = ents[pick_name].get_pos().cpu().numpy()
    log.log("ManipPipeline", f"  PLACE done",
            error_cm=f"{err*100:.1f}",
            place_ms=f"{place_ms:.0f}")
    log.log("ManipPipeline", f"  final {pick_name}: [{cube_final[0]:.3f}, {cube_final[1]:.3f}, {cube_final[2]:.3f}]")

    log.log("ActionScheduler", "All actions complete", progress=str(scheduler.get_progress()))

    # ═════════════════════════════════════════════════════════
    # MODULE 8: SceneMemory (update + verify)
    # ═════════════════════════════════════════════════════════
    log.log("SceneMemory", "Updating positions after manipulation...")
    memory.update_positions(ents)
    memory.record_action("pick", pick_name)
    memory.record_action("place", pick_name, target_name="blue_cube")

    # Find nearest object to verify placement
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
                target=nearest,
                success=verify_result["success"],
                distance=f"{verify_result['distance_to_target']:.3f}m",
                moved=verify_result["moved_from_start"])
    else:
        log.log("SceneMemory", "No reference object for verification")

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

    # Check if pick was a failure
    grasp_ok = lifted and (post_pick_pos[pick_name][2] > pre_pick_pos[pick_name][2] + 0.05)
    log.log("FailDetector", "  Grasp check",
            passed=grasp_ok,
            z_before=f"{pre_pick_pos[pick_name][2]:.3f}",
            z_after=f"{post_pick_pos[pick_name][2]:.3f}")

    # Check if placement was a failure
    place_ok = err < 0.15
    log.log("FailDetector", "  Placement check",
            passed=place_ok,
            error=f"{err*100:.1f}cm",
            threshold="15cm")

    # Check other objects disturbed
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
    # OUTPUT: Video + Comparison
    # ═════════════════════════════════════════════════════════
    log.log("Output", "Saving video and comparison...")
    for _ in range(30): scene.step(2); vis_cam.render()
    vis_cam.stop_recording(save_to_filename="demo/output/test_e2e.mp4", fps=30)

    import cv2
    h, w = img_before.shape[:2]
    canvas = np.zeros((h + 50, w * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = img_before
    canvas[:h, w:] = img_after
    cv2.rectangle(canvas, (0, h), (w * 2, h + 50), (30, 30, 30), -1)
    cv2.putText(canvas, f"Before | VLM: {vlm_ms:.0f}ms", (10, h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"After | Place: {err*100:.1f}cm error", (w + 10, h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    cv2.line(canvas, (w, 0), (w, h), (100, 100, 100), 2)
    Image.fromarray(canvas).save("demo/output/test_comparison.png")
    log.log("Output", "Files saved: test_before.png, test_after.png, test_comparison.png, test_e2e.mp4")

    # ═════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════
    total_ms = pick_ms + place_ms + vlm_ms
    log.summary()

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  VLM:     Qwen3-VL-8B via vLLM | {vlm_ms:.0f}ms")
    print(f"  Pick:    {pick_name} | lifted={lifted} | {pick_ms:.0f}ms")
    print(f"  Place:   error {err*100:.1f}cm | {place_ms:.0f}ms")
    print(f"  Verify:  pixel_change={verify_cam['success']} | grasp_ok={grasp_ok} | place_ok={place_ok}")
    print(f"  Objects: {len(disturbed)} disturbed")
    print(f"  Total:   {total_ms:.0f}ms pipeline time")
    print(f"  Status:  {overall}")
    print(f"{'='*70}")

    # Print full JSON log
    print(f"\n  Full step log ({len(log.steps)} entries):")
    for s in log.steps:
        print(f"    {s}")


if __name__ == "__main__":
    main()
