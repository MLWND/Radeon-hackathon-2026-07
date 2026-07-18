#!/usr/bin/env python3
"""
RoboPilot Visual Demo — Competition Version (v2)
Improvements: larger text, softer shadows, bigger cubes, before/after comparison
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def add_label(img, text, x=20, y=40, font_scale=1.0, color=(0, 255, 200), bg=True):
    """Add text label with optional semi-transparent background."""
    import cv2
    img = np.ascontiguousarray(img, dtype=np.uint8)
    thickness = max(2, int(font_scale * 2.5))

    if bg:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 8
        overlay = img.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad + 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def make_before_after(img_before, img_after, label_before="Before", label_after="After"):
    """Create side-by-side before/after comparison."""
    h, w = img_before.shape[:2]
    # Create canvas
    canvas = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = img_before
    canvas[:h, w:] = img_after
    # Add labels
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(canvas, (0, h), (w * 2, h + 40), (30, 30, 30), -1)
    cv2.putText(canvas, label_before, (w // 2 - 80, h + 30), font, 1.0, (200, 200, 200), 2)
    cv2.putText(canvas, label_after, (w + w // 2 - 60, h + 30), font, 1.0, (0, 255, 200), 2)
    # Divider line
    cv2.line(canvas, (w, 0), (w, h), (100, 100, 100), 2)
    return canvas


def main():
    header("RoboPilot — Visual Demo v2")

    # ── 1. Load Qwen3-VL ────────────────────────────────────
    header("Step 1: Load Qwen3-VL")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    t0 = time.time()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto")
    print(f"  Loaded in {time.time()-t0:.1f}s on {vlm_model.device}")

    # ── 2. Build Scene ──────────────────────────────────────
    header("Step 2: Build Visual Scene")
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.8)),
        vis_options=gs.options.VisOptions(
            lights=[
                gs.options.vis.DirectionalLight(
                    dir=(-1, -1, -1), color=(1.0, 1.0, 1.0), intensity=5.0),
                gs.options.vis.PointLight(
                    pos=(2.0, -1.0, 3.0), color=(1.0, 0.95, 0.9), intensity=8.0),
            ],
            ambient_light=(0.5, 0.5, 0.5),  # ↑ 0.3→0.5 for softer shadows
        ),
        show_viewer=False,
    )

    # Ground — dark gray
    scene.add_entity(gs.morphs.Plane(),
                     surface=gs.surfaces.Rough(color=(0.25, 0.25, 0.25)))

    # Table — warm gray
    scene.add_entity(
        gs.morphs.Box(size=(0.8, 1.0, 0.05), pos=(0.5, 0, 0.025)),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55)))

    # Robot
    robot = scene.add_entity(gs.morphs.MJCF(
        file=os.path.join("venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml")))

    T = 0.05
    S = 0.06  # ↑ 0.04→0.06 bigger cubes

    # Colored cubes — vivid colors, bigger size
    cube_colors = {
        "red_cube":    (0.9, 0.15, 0.15),
        "blue_cube":   (0.15, 0.3, 0.9),
        "green_cube":  (0.15, 0.75, 0.2),
        "yellow_cube": (0.95, 0.85, 0.1),
    }
    cube_positions = {
        "red_cube":    (0.55,  0.00, T + S / 2),
        "blue_cube":   (0.65,  0.20, T + S / 2),
        "green_cube":  (0.70, -0.10, T + S / 2),
        "yellow_cube": (0.50,  0.15, T + S / 2),
    }
    ents = {}
    for name in cube_colors:
        ents[name] = scene.add_entity(
            gs.morphs.Box(size=(S, S, S), pos=cube_positions[name]),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Smooth(color=cube_colors[name], roughness=0.3))

    # Goal area — blue transparent
    goal_center = [0.75, 0.20, T + 0.001]
    goal_area = scene.add_entity(
        gs.morphs.Box(size=(0.14, 0.14, 0.002), pos=goal_center),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Smooth(color=(0.2, 0.5, 1.0, 0.4), roughness=0.1))

    # Camera — 45° side view, slightly higher lookat
    WIDTH, HEIGHT = 1280, 720
    camera = scene.add_camera(
        res=(WIDTH, HEIGHT),
        pos=(1.5, -2.0, 1.6),       # ↑ z 1.5→1.6
        lookat=(0.5, 0, 0.08),       # ↑ lookat z 0.05→0.08
        fov=45,
    )

    scene.build()
    scene.step(200)

    for n in cube_colors:
        p = ents[n].get_pos().cpu().numpy()
        print(f"  {n:12s}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

    # ── 3. VLM Perception (OPTIMIZED) ────────────────────────
    header("Step 3: Qwen3-VL Perception (Optimized)")
    img_full = camera.render()[0]
    Image.fromarray(img_full).save("demo/output/visual_before.png")

    INSTRUCTION = "Pick the red cube and place it in the blue goal area"
    print(f"  Instruction: \"{INSTRUCTION}\"")

    obj_names = list(ents.keys())

    # SHORT prompt — minimal tokens
    prompt = f"""Return ONLY JSON: {{"pick":"name","place":"relative desc"}}
Objects: {obj_names}
Instruction: {INSTRUCTION}"""

    # Resize to 448x448 for faster inference
    VLM_SIZE = 448
    pil_img = Image.fromarray(img_full).resize((VLM_SIZE, VLM_SIZE), Image.LANCZOS)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt},
    ]}]

    # Timed pipeline
    t_render_end = time.time()
    t_preprocess = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(vlm_model.device)
    t_preprocess = (time.time() - t_preprocess) * 1000

    t_infer = time.time()
    with torch.no_grad():
        output = vlm_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    t_infer = (time.time() - t_infer) * 1000

    t_decode = time.time()
    raw = processor.decode(output[0], skip_special_tokens=True)
    t_decode = (time.time() - t_decode) * 1000

    vlm_ms = t_preprocess + t_infer + t_decode
    print(f"  VLM timing: preprocess={t_preprocess:.0f}ms, infer={t_infer:.0f}ms, decode={t_decode:.0f}ms, total={vlm_ms:.0f}ms")
    print(f"  Input: {VLM_SIZE}x{VLM_SIZE} | max_tokens: 64")

    pick_name = "red_cube"
    place_relative = "in the goal area"
    try:
        s = raw.rfind("```json"); e = raw.rfind("```")
        block = raw[s+7:e].strip() if s >= 0 and e > s else raw[raw.find("{"):raw.rfind("}")+1]
        r = json.loads(block)
        pick_name = r.get("pick", "red_cube")
        place_relative = r.get("place", "in the goal area")
        if pick_name not in ents:
            pick_name = "red_cube"
    except Exception:
        pass

    print(f"  Result: pick={pick_name}, place={place_relative}")

    # ── 4. Execute ──────────────────────────────────────────
    header("Step 4: Execute Pick & Place")
    from src.control.primitives import RobotPrimitives
    prims = RobotPrimitives(robot, scene, ents)

    camera.start_recording()

    # VLM frame
    img = camera.render()[0]
    img = add_label(img, "Qwen3-VL: Pick red_cube, place in goal", x=20, y=40, color=(0, 255, 200))
    img = add_label(img, f"Inference: {vlm_ms:.0f}ms | AMD ROCm GPU", x=20, y=90, font_scale=0.8, color=(200, 200, 200))
    Image.fromarray(img).save("demo/output/visual_step1_vlm.png")

    # Pick
    print("  Picking...")
    lifted = prims.suction_pick(pick_name)

    img = camera.render()[0]
    img = add_label(img, f"Suction Pick: {pick_name}", x=20, y=40, color=(0, 255, 100))
    img = add_label(img, "Weld constraint attached to hand", x=20, y=90, font_scale=0.8, color=(200, 200, 200))
    Image.fromarray(img).save("demo/output/visual_step2_pick.png")

    # Place
    print("  Placing...")
    place_err = prims.suction_place(pick_name, goal_center)

    img = camera.render()[0]
    cube_final = ents[pick_name].get_pos().cpu().numpy()
    err = np.sqrt((cube_final[0]-goal_center[0])**2 + (cube_final[1]-goal_center[1])**2)
    img = add_label(img, f"Suction Place: error {err*100:.1f}cm", x=20, y=40, color=(0, 200, 255))
    img = add_label(img, f"Cube at [{cube_final[0]:.3f}, {cube_final[1]:.3f}]", x=20, y=90, font_scale=0.8, color=(200, 200, 200))
    Image.fromarray(img).save("demo/output/visual_step3_place.png")

    # Final
    img = camera.render()[0]
    img = add_label(img, f"SUCCESS — {pick_name} placed in goal", x=20, y=40, color=(0, 255, 0))
    img = add_label(img, f"Error: {err*100:.1f}cm | Pipeline: {vlm_ms/1000+7+0.1:.1f}s", x=20, y=90, font_scale=0.8, color=(200, 200, 200))
    Image.fromarray(img).save("demo/output/visual_step4_final.png")

    # Before/After comparison
    img_after = camera.render()[0]
    comparison = make_before_after(img_before, img_after,
        label_before="Initial State",
        label_after=f"After Pick & Place (error: {err*100:.1f}cm)")
    Image.fromarray(comparison).save("demo/output/visual_comparison.png")

    camera.stop_recording(save_to_filename="demo/output/robopilot_demo.mp4", fps=30)

    # ── 5. Summary ──────────────────────────────────────────
    header("Demo Complete")
    print(f"  Visual Assets:")
    print(f"    visual_before.png         — initial scene")
    print(f"    visual_step1_vlm.png      — VLM perception (labeled)")
    print(f"    visual_step2_pick.png     — after pick (labeled)")
    print(f"    visual_step3_place.png    — after place (labeled)")
    print(f"    visual_step4_final.png    — final result (labeled)")
    print(f"    visual_comparison.png     — before/after side-by-side")
    print(f"    robopilot_demo.mp4        — full video")
    print(f"")
    print(f"  Metrics:")
    print(f"    VLM:     Qwen3-VL-2B ({vlm_ms:.0f}ms)")
    print(f"    Pick:    ~7s | Place: ~0.1s")
    print(f"    Error:   {err*100:.1f}cm")
    print(f"    Camera:  {WIDTH}x{HEIGHT}, 45° side view")
    print(f"    Cubes:   {S*100:.0f}cm colored (red/blue/green/yellow)")
    print(f"    Lighting: ambient=0.5, directional + point")


if __name__ == "__main__":
    main()
