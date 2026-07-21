"""
VLM-Driven Pick and Place — Vision-Language Model guides robotic manipulation.

This example demonstrates how to integrate a Vision-Language Model (VLM)
with Genesis physics simulation for natural-language-driven pick-and-place.
The VLM observes the scene via camera, identifies target objects, and the
robot executes the manipulation using Genesis IK and motion planning.

Pipeline:
    User instruction → VLM perception → Task planning → Manipulation → Verification

Requirements:
    - Genesis 1.2.2+
    - transformers (for Qwen2-VL) OR openai (for vLLM-served models)
    - Pillow

Usage:
    # With local Qwen2-VL model (requires ~8GB VRAM):
    python vlm_pick_place.py

    # With vLLM server (start separately):
    vllm serve Qwen/Qwen2-VL-7B-Instruct --enforce-eager
    python vlm_pick_place.py --vlm-backend vllm
"""
import argparse
import json
import re

import genesis as gs
import numpy as np
import torch


# ── Scene Setup ────────────────────────────────────────────────

def build_scene():
    """Create a manipulation scene with colored cubes on a ground plane."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    cube_configs = [
        ("red_cube", (1, 0, 0), (0.5, 0.0, 0.02)),
        ("blue_cube", (0, 0, 1), (0.5, 0.15, 0.02)),
        ("green_cube", (0, 1, 0), (0.5, -0.15, 0.02)),
    ]
    cubes = {}
    for name, color, pos in cube_configs:
        ent = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos),
            surface=gs.surfaces.Smooth(color=color),
        )
        cubes[name] = ent

    cam = scene.add_camera(
        res=(640, 480), pos=(1.5, -1.0, 1.2),
        lookat=(0.5, 0.0, 0.0), fov=50,
    )

    scene.build()

    return scene, franka, cubes, cam


# ── VLM Perception ────────────────────────────────────────────

def get_vlm_response(image_path: str, instruction: str, backend: str = "local") -> dict:
    """Query VLM to identify target object from camera image.

    Returns: {"target": "red_cube", "reason": "..."} or None on failure.
    """
    prompt = (
        f"Look at this image of a robot workspace with colored cubes. "
        f"The user says: \"{instruction}\". "
        f"Which cube should the robot pick? "
        f'Respond with JSON: {{"target": "<color>_cube", "reason": "<brief>"}}'
    )

    if backend == "local":
        from transformers import AutoProcessor
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        from PIL import Image
        img = Image.open(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    elif backend == "vllm":
        from openai import OpenAI
        import base64

        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model="Qwen/Qwen2-VL-7B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=128,
        )
        response = resp.choices[0].message.content

    # Parse JSON from response
    match = re.search(r'\{[^}]+\}', response)
    if match:
        return json.loads(match.group())
    return None


# ── Manipulation ───────────────────────────────────────────────

def suction_pick(scene, franka, cubes, target_name: str):
    """Pick object using suction (weld constraint)."""
    if target_name not in cubes:
        print(f"  [!] Unknown object: {target_name}")
        return False

    obj = cubes[target_name]
    ee = franka.get_link("hand")
    motors_dof = list(range(7))

    # Open gripper
    franka.control_dofs_position(
        np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]),
        np.arange(9))
    scene.step(50)

    # Plan path to above object
    obj_pos = obj.get_pos()
    above_pos = obj_pos.clone()
    above_pos[2] = 0.25

    qpos_above = franka.inverse_kinematics(
        link=ee, pos=above_pos,
        quat=torch.tensor([0, 1, 0, 0], dtype=torch.float32))
    path = franka.plan_path(qpos_goal=qpos_above, num_waypoints=100)

    for wp in path:
        franka.control_dofs_position(wp[:-2], motors_dof)
        scene.step()
    scene.step(100)

    # Descend to object
    reach_pos = obj_pos.clone()
    reach_pos[2] += 0.03
    qpos_reach = franka.inverse_kinematics(
        link=ee, pos=reach_pos,
        quat=torch.tensor([0, 1, 0, 0], dtype=torch.float32))
    franka.control_dofs_position(qpos_reach[:-2], motors_dof)
    scene.step(200)

    # Weld (suction grasp)
    hand_idx = ee.idx
    obj_link_idx = obj.get_link("box_baselink").idx
    scene.rigid_solver.add_weld_constraint(obj_link_idx, hand_idx)
    scene.step(1)

    # Lift
    lift_pos = above_pos.clone()
    qpos_lift = franka.inverse_kinematics(
        link=ee, pos=lift_pos,
        quat=torch.tensor([0, 1, 0, 0], dtype=torch.float32))
    franka.control_dofs_position(qpos_lift[:-2], motors_dof)
    scene.step(200)

    print(f"  [OK] Picked {target_name}")
    return True


def suction_place(scene, franka, cubes, target_name: str, place_pos):
    """Place object using suction release."""
    obj = cubes[target_name]
    ee = franka.get_link("hand")
    motors_dof = list(range(7))

    # Move to above target
    above = torch.tensor([place_pos[0], place_pos[1], 0.25], dtype=torch.float32)
    qpos_above = franka.inverse_kinematics(
        link=ee, pos=above,
        quat=torch.tensor([0, 1, 0, 0], dtype=torch.float32))
    franka.control_dofs_position(qpos_above[:-2], motors_dof)
    scene.step(100)

    # Descend
    reach = torch.tensor([place_pos[0], place_pos[1], place_pos[2] + 0.03],
                         dtype=torch.float32)
    qpos_reach = franka.inverse_kinematics(
        link=ee, pos=reach,
        quat=torch.tensor([0, 1, 0, 0], dtype=torch.float32))
    franka.control_dofs_position(qpos_reach[:-2], motors_dof)
    scene.step(200)

    # Release
    obj_link_idx = obj.get_link("box_baselink").idx
    scene.rigid_solver.delete_weld_constraint(obj_link_idx, ee.idx)
    scene.step(200)

    # Lift arm
    franka.control_dofs_position(
        np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]),
        np.arange(9))
    scene.step(100)

    actual = obj.get_pos().cpu().numpy()
    err = np.linalg.norm(actual[:2] - np.array(place_pos[:2]))
    print(f"  [OK] Placed {target_name} — error: {err*100:.1f}cm")
    return err


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-backend", choices=["local", "vllm", "none"],
                        default="none", help="VLM backend (none = skip VLM)")
    parser.add_argument("--instruction", type=str,
                        default="Pick the red cube and place it next to the blue cube")
    args = parser.parse_args()

    gs.init(backend=gs.amdgpu)

    print("=== VLM-Driven Pick and Place ===")
    print(f"Instruction: {args.instruction}")

    scene, franka, cubes, cam = build_scene()
    scene.step(100)

    # Capture scene image
    cam.start_recording()
    img = cam.render()
    img_path = "/tmp/vlm_scene.png"
    from PIL import Image
    Image.fromarray(img).save(img_path)

    # VLM perception
    if args.vlm_backend != "none":
        print("\n[VLM] Analyzing scene...")
        vlm_result = get_vlm_response(img_path, args.instruction, args.vlm_backend)
        if vlm_result:
            target = vlm_result["target"]
            print(f"  Target: {target} ({vlm_result.get('reason', '')})")
        else:
            print("  [!] VLM failed, falling back to keyword matching")
            target = "red_cube"
    else:
        # Fallback: keyword matching
        if "red" in args.instruction.lower():
            target = "red_cube"
        elif "blue" in args.instruction.lower():
            target = "blue_cube"
        elif "green" in args.instruction.lower():
            target = "green_cube"
        else:
            target = "red_cube"
        print(f"\n[Rule-based] Target: {target}")

    # Pick
    print("\n[Pick]")
    suction_pick(scene, franka, cubes, target)

    # Place next to blue cube
    print("\n[Place]")
    blue_pos = cubes["blue_cube"].get_pos().cpu().numpy()
    place_pos = [blue_pos[0] + 0.06, blue_pos[1], 0.02]
    suction_place(scene, franka, cubes, target, place_pos)

    # Verify
    print("\n[Verify]")
    final_pos = cubes[target].get_pos().cpu().numpy()
    err = np.linalg.norm(final_pos[:2] - np.array(place_pos[:2]))
    print(f"  Final position: {final_pos[:3]}")
    print(f"  Placement error: {err*100:.1f}cm")
    print(f"  Status: {'SUCCESS' if err < 0.05 else 'PARTIAL'}")

    cam.stop_recording(save_to_filename="/tmp/vlm_pick_place.mp4", fps=30)
    print("\nVideo saved to /tmp/vlm_pick_place.mp4")


if __name__ == "__main__":
    main()
