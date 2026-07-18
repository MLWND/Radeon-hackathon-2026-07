#!/usr/bin/env python3
"""
RoboPilot Full Demo: Qwen3-VL + Suction Pick & Place
Complete end-to-end pipeline on AMD ROCm GPU.
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    header("RoboPilot — Qwen3-VL + Suction Pick & Place Demo")

    # ── 1. Load Qwen3-VL ────────────────────────────────────
    header("Step 1: Load Qwen3-VL")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    t0 = time.time()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto")
    print(f"  Model: Qwen/Qwen3-VL-2B-Instruct")
    print(f"  Device: {vlm_model.device}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # ── 2. Build Scene ──────────────────────────────────────
    header("Step 2: Build Genesis Scene")
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.8)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())
    scene.add_entity(gs.morphs.Box(size=(0.8, 1.0, 0.05), pos=(0.5, 0, 0.025)),
                     material=gs.materials.Kinematic())
    robot = scene.add_entity(gs.morphs.MJCF(
        file=os.path.join("venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml")))

    T, S = 0.05, 0.04
    obj_specs = {
        "red_cube":    (0.60,  0.00, T + S / 2),
        "blue_cube":   (0.65,  0.20, T + S / 2),
        "green_cube":  (0.70, -0.10, T + S / 2),
        "yellow_cube": (0.55,  0.15, T + S / 2),
    }
    ents = {}
    for name, pos in obj_specs.items():
        ents[name] = scene.add_entity(
            gs.morphs.Box(size=(S, S, S), pos=pos),
            material=gs.materials.Rigid())

    camera = scene.add_camera(
        res=(640, 480), pos=(0.3, -1.2, 1.6),
        lookat=(0.3, 0, 0.05), fov=55)
    scene.build()
    scene.step(200)

    print(f"  Table: Kinematic box at z={T}")
    print(f"  Robot: Franka Panda (MJCF)")
    for n in obj_specs:
        p = ents[n].get_pos().cpu().numpy()
        print(f"  {n:12s}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

    # ── 3. Camera Before ────────────────────────────────────
    header("Step 3: Camera Capture (Before)")
    img_before = camera.render()[0]
    Image.fromarray(img_before).save("demo/output/before.png")
    pos_before = {n: ents[n].get_pos().cpu().numpy().tolist() for n in ents}
    print(f"  Saved: demo/output/before.png")

    # ── 4. VLM Perception ──────────────────────────────────
    header("Step 4: Qwen3-VL Perception")
    INSTRUCTION = "Pick the red cube and place it next to the blue cube"
    print(f"  Instruction: \"{INSTRUCTION}\"")

    prompt = f"""Analyze this image for a robotic pick-and-place task.
Instruction: {INSTRUCTION}
Identify the object to PICK and where to PLACE it.
Output ONLY valid JSON:
{{"pick": "object_name", "place_xyz": [x, y, z], "reasoning": "brief explanation"}}
Available objects: {list(ents.keys())}
Table surface at z={T}, objects at z~{T + S / 2:.3f}
Robot workspace: x=[0.3, 0.8], y=[-0.3, 0.3]"""

    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.fromarray(img_before)},
        {"type": "text", "text": prompt},
    ]}]

    t0 = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.fromarray(img_before)],
                       return_tensors="pt").to(vlm_model.device)
    with torch.no_grad():
        output = vlm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.decode(output[0], skip_special_tokens=True)
    vlm_ms = (time.time() - t0) * 1000

    pick_name, place_xyz, reasoning = "red_cube", [0.75, 0.2, T + S / 2], ""
    try:
        s = raw.rfind("```json")
        e = raw.rfind("```")
        block = raw[s + 7:e].strip() if s >= 0 and e > s else raw[raw.find("{"):raw.rfind("}") + 1]
        r = json.loads(block)
        pick_name = r.get("pick", "red_cube")
        place_xyz = r.get("place_xyz", place_xyz)
        reasoning = r.get("reasoning", "")
        if pick_name not in ents:
            pick_name = "red_cube"
    except Exception:
        reasoning = "parse failed, using default"

    print(f"  Inference: {vlm_ms:.0f}ms")
    print(f"  ┌─ VLM Decision ──────────────────────")
    print(f"  │ Pick:  {pick_name}")
    print(f"  │ Place: [{place_xyz[0]:.3f}, {place_xyz[1]:.3f}, {place_xyz[2]:.3f}]")
    print(f"  │ Why:   {reasoning[:100]}")
    print(f"  └─────────────────────────────────────")

    # ── 5. Suction Pick ─────────────────────────────────────
    header("Step 5: Suction Pick")
    from src.control.primitives import RobotPrimitives
    prims = RobotPrimitives(robot, scene, ents)

    obj_pos = ents[pick_name].get_pos().cpu().numpy()
    print(f"  Target: {pick_name} at [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")

    t0 = time.time()
    lifted = prims.suction_pick(pick_name)
    pick_t = time.time() - t0

    cube = ents[pick_name].get_pos().cpu().numpy()
    print(f"  Time: {pick_t:.1f}s | Cube: [{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}] | Lifted: {lifted}")

    # ── 6. Suction Place ────────────────────────────────────
    header("Step 6: Suction Place")
    print(f"  Target: [{place_xyz[0]:.3f}, {place_xyz[1]:.3f}, {place_xyz[2]:.3f}]")

    t0 = time.time()
    prims.suction_place(pick_name, place_xyz)
    place_t = time.time() - t0

    cube_f = ents[pick_name].get_pos().cpu().numpy()
    print(f"  Time: {place_t:.1f}s | Cube: [{cube_f[0]:.3f},{cube_f[1]:.3f},{cube_f[2]:.3f}]")

    # ── 7. Camera Verify ────────────────────────────────────
    header("Step 7: Camera Verification")
    img_after = camera.render()[0]
    Image.fromarray(img_after).save("demo/output/after.png")

    print(f"\n  Position Comparison:")
    print(f"  {'Object':12s} | {'Before':>20s} | {'After':>20s} | {'Delta':>8s}")
    print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*20}-+-{'-'*8}")
    for n in ents:
        pb = pos_before[n]
        pa = ents[n].get_pos().cpu().tolist()
        delta = np.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2 + (pa[2] - pb[2])**2)
        moved = "*" if delta > 0.01 else " "
        print(f"  {n:12s} | [{pb[0]:.3f},{pb[1]:.3f},{pb[2]:.3f}] | [{pa[0]:.3f},{pa[1]:.3f},{pa[2]:.3f}] | {delta*100:5.1f}cm {moved}")

    diff = np.abs(img_before.astype(float) - img_after.astype(float)).mean()
    print(f"\n  Image pixel diff: {diff:.1f}/255 ({'changed' if diff > 5 else 'minimal'})")
    print(f"  Saved: demo/output/after.png")

    # ── 8. Summary ──────────────────────────────────────────
    header("Demo Complete")
    placement_err = np.sqrt(
        (cube_f[0] - place_xyz[0])**2 + (cube_f[1] - place_xyz[1])**2)

    print(f"  Pipeline:")
    print(f"    Qwen3-VL → Task Plan → OMPL → Suction Pick → Place → Camera Verify")
    print(f"")
    print(f"  Results:")
    print(f"    VLM:       Qwen3-VL-2B (native Qwen3VLForConditionalGeneration)")
    print(f"    VLM time:  {vlm_ms:.0f}ms")
    print(f"    Pick time: {pick_t:.1f}s")
    print(f"    Place time:{place_t:.1f}s")
    print(f"    Error:     {placement_err*100:.1f}cm")
    print(f"    Backend:   Genesis 1.2.2 + gs.amdgpu (AMD ROCm 7.2)")
    print(f"    GPU:       AMD Radeon Graphics, 48GB VRAM")
    print(f"    Status:    {'SUCCESS' if placement_err < 0.15 else 'NEEDS TUNING'}")


if __name__ == "__main__":
    main()
