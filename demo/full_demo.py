#!/usr/bin/env python3
"""
RoboPilot Full Demo — Official Genesis Pattern
Rebuilt from: grasp_env.py + suction_cup.py + IK_motion_planning_grasp.py
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys, logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    header("RoboPilot — Official Genesis Pattern")

    # ═══ Step 1: Load Qwen3-VL ══════════════════════════════
    header("Step 1: Load Qwen3-VL")
    from openai import OpenAI
    from PIL import Image
    import base64, io
    vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=60)

    # ═══ Step 2: Build Scene (official pattern) ═════════════
    header("Step 2: Build Scene (Official)")
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )

    # Ground plane ONLY — official pattern
    scene.add_entity(gs.morphs.Plane())

    # Robot with collision vis mode (official)
    robot = scene.add_entity(
        gs.morphs.MJCF(file=os.path.join(
            "venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml")),
        vis_mode="collision",
    )

    # Cubes on ground — official z=0.02
    ents = {}
    ents["red_cube"] = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)))
    ents["blue_cube"] = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.4, 0.2, 0.02)),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)))
    ents["green_cube"] = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.7, -0.1, 0.02)),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)))

    # Camera
    camera = scene.add_camera(res=(1280, 720), pos=(1.5, -2.0, 1.6),
                              lookat=(0.5, 0, 0.0), fov=45)
    scene.build()
    scene.step(200)

    # ═══ Step 3: VLM Perception ═════════════════════════════
    header("Step 3: Qwen3-VL Perception")
    img_before = camera.render()[0]
    Image.fromarray(img_before).save("demo/output/visual_before.png")

    pil_img = Image.fromarray(img_before)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    INSTRUCTION = "Pick the red cube and place it next to the blue cube"

    t0 = time.time()
    response = vlm_client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": f'Return ONLY JSON: {{"pick":"name"}}\nObjects: {list(ents.keys())}\nInstruction: {INSTRUCTION}'}
        ]}],
        max_tokens=32,
    )
    vlm_ms = (time.time() - t0) * 1000
    raw = response.choices[0].message.content

    pick_name = "red_cube"
    try:
        s = raw.rfind("{"); e = raw.rfind("}") + 1
        pick_name = json.loads(raw[s:e]).get("pick", "red_cube")
        if pick_name not in ents:
            pick_name = "red_cube"
    except Exception:
        pass
    print(f"  VLM ({vlm_ms:.0f}ms): pick={pick_name}")

    # ═══ Step 4: Manipulation (official suction_cup.py) ══════
    header("Step 4: Manipulation (Official)")
    from src.control.primitives import ManipulationPipeline
    pipe = ManipulationPipeline(robot, scene, ents)

    camera.start_recording()
    for _ in range(20): scene.step(1); camera.render()

    # Pick
    print("  Picking...")
    lifted = pipe.suction_pick(pick_name)

    # Place at goal
    goal = [0.4, 0.2, 0.02]
    print("  Placing...")
    err = pipe.suction_place(pick_name, goal)

    # ═══ Step 5: Verification ══════════════════════════════
    header("Step 5: Verification")
    img_after = camera.render()[0]
    Image.fromarray(img_after).save("demo/output/visual_after.png")

    # Before/After comparison
    import cv2
    h, w = img_before.shape[:2]
    canvas = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = img_before
    canvas[:h, w:] = img_after
    cv2.rectangle(canvas, (0, h), (w * 2, h + 40), (30, 30, 30), -1)
    cv2.putText(canvas, "Before", (w // 2 - 40, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(canvas, "After", (w + w // 2 - 30, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
    cv2.line(canvas, (w, 0), (w, h), (100, 100, 100), 2)
    Image.fromarray(canvas).save("demo/output/visual_comparison.png")

    for _ in range(30): scene.step(2); camera.render()
    camera.stop_recording(save_to_filename="demo/output/robopilot_demo.mp4", fps=30)

    # ═══ Step 6: Summary ═══════════════════════════════════
    header("Results")
    cube_final = ents[pick_name].get_pos().cpu().numpy()
    print(f"  Pipeline: Official Genesis pattern")
    print(f"  VLM:      Qwen3-VL-8B ({vlm_ms:.0f}ms)")
    print(f"  Pick:     {lifted}")
    print(f"  Place:    {err*100:.1f}cm error")
    print(f"  Cube:     [{cube_final[0]:.3f}, {cube_final[1]:.3f}, {cube_final[2]:.3f}]")
    print(f"  Status:   {'SUCCESS' if err < 0.10 else 'NEEDS WORK'}")


if __name__ == "__main__":
    main()
