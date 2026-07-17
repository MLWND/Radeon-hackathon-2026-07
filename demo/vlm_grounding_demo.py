"""
VLM Grounding Demo — Hybrid Approach
Qwen3-VL for task understanding + Genesis for precise coordinates

Usage:
    python demo/vlm_grounding_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import json
import numpy as np
from PIL import Image


# ── VLM Module ───────────────────────────────────────────────

_vlm_cache = {"proc": None, "model": None}

def load_vlm():
    if _vlm_cache["proc"] is not None:
        return _vlm_cache["proc"], _vlm_cache["model"]

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading {model_name}...")
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto",
    )
    print(f"VLM loaded on {model.device}")
    _vlm_cache["proc"] = proc
    _vlm_cache["model"] = model
    return proc, model


def vlm_understand(proc, model, image: np.ndarray, instruction: str) -> dict:
    import torch
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(image)
    prompt = f"""Task: {instruction}

What object to pick? What is the target?
Reply in JSON: {{"pick":"object description","place":"target description","reasoning":"brief"}}"""

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt},
    ]}]

    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil_img], return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = (time.time() - start) * 1000

    response = proc.decode(output[0], skip_special_tokens=True)
    return _parse_vlm_output(response, elapsed)


def _parse_vlm_output(response: str, elapsed_ms: float) -> dict:
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
            result["inference_time_ms"] = elapsed_ms
            return result
    except json.JSONDecodeError:
        pass
    return {"pick": "unknown", "place": "unknown", "reasoning": "parse failed", "inference_time_ms": elapsed_ms}


# ── Object Registry ──────────────────────────────────────────

# Known objects in the tabletop scene (from scene_manager)
KNOWN_OBJECTS = {
    "red cup": {"type": "cup", "color": "red", "entity": "red_cup"},
    "cup": {"type": "cup", "color": "red", "entity": "red_cup"},
    "blue box": {"type": "box", "color": "blue", "entity": "blue_box"},
    "box": {"type": "box", "color": "blue", "entity": "blue_box"},
    "apple": {"type": "apple", "color": "red", "entity": "apple"},
    "bottle": {"type": "bottle", "color": "clear", "entity": "bottle"},
}


def resolve_object(vlm_output: dict, scene_objects: dict) -> dict:
    """Match VLM text output to actual Genesis object positions."""
    pick_desc = vlm_output.get("pick", "").lower()
    place_desc = vlm_output.get("place", "").lower()

    pick_obj = None
    place_obj = None

    for desc, info in KNOWN_OBJECTS.items():
        if desc in pick_desc and info["entity"] in scene_objects:
            pick_obj = {
                "name": info["entity"],
                "type": info["type"],
                "color": info["color"],
                "position": scene_objects[info["entity"]].get_pos().tolist(),
            }
        if desc in place_desc and info["entity"] in scene_objects:
            place_obj = {
                "name": info["entity"],
                "type": info["type"],
                "color": info["color"],
                "position": scene_objects[info["entity"]].get_pos().tolist(),
            }

    # Fallback: try direct name matching
    if not pick_obj:
        for name, obj in scene_objects.items():
            if any(kw in pick_desc for kw in name.split("_")):
                pick_obj = {"name": name, "type": name, "color": "unknown", "position": obj.get_pos().tolist()}
                break
    if not place_obj:
        for name, obj in scene_objects.items():
            if any(kw in place_desc for kw in name.split("_")):
                place_obj = {"name": name, "type": name, "color": "unknown", "position": obj.get_pos().tolist()}
                break

    return {
        "task": "pick_place",
        "reasoning": vlm_output.get("reasoning", ""),
        "object": pick_obj or {"name": "red_cup", "type": "cup", "color": "red", "position": scene_objects.get("red_cup", type("", (), {"get_pos": lambda: type("", (), {"tolist": lambda: [0.2, 0.15, 0.43]})()})()).get_pos().tolist()},
        "target": place_obj or {"name": "blue_box", "type": "box", "color": "blue", "position": scene_objects.get("blue_box", type("", (), {"get_pos": lambda: type("", (), {"tolist": lambda: [-0.2, 0.15, 0.42]})()})()).get_pos().tolist()},
        "inference_time_ms": vlm_output.get("inference_time_ms", 0),
    }


# ── Main Demo ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  VLM Grounding Demo — Hybrid Approach")
    print("  Qwen3-VL (understanding) + Genesis (coordinates)")
    print("=" * 60)

    # Phase 1: Build Scene
    print("\n[1/5] Building tabletop scene...")
    from src.sim.scene_manager import create_tabletop_scene
    scene = create_tabletop_scene(show_viewer=False)
    scene.settle(100)

    # Phase 2: Render Image
    print("\n[2/5] Rendering camera image...")
    rgb = scene.render_rgb()
    Image.fromarray(rgb).save("demo/output/vlm_input.png")
    print(f"  Image: {rgb.shape}")

    # Phase 3: VLM Understanding
    print("\n[3/5] Running Qwen3-VL understanding...")
    proc, model = load_vlm()

    instructions = [
        "Pick up the red cup and place it in the blue box",
        "Move the apple next to the bottle",
    ]

    for i, instruction in enumerate(instructions):
        print(f"\n  Instruction {i+1}: {instruction}")
        vlm_out = vlm_understand(proc, model, rgb, instruction)
        print(f"  VLM output ({vlm_out.get('inference_time_ms', 0):.0f}ms): {vlm_out}")

        # Phase 4: Resolve to Genesis coordinates
        print(f"\n[4/5] Resolving to Genesis coordinates...")
        task = resolve_object(vlm_out, scene.objects)
        print(f"  Pick: {task['object']['name']} at {[f'{x:.2f}' for x in task['object']['position']]}")
        print(f"  Place: {task['target']['name']} at {[f'{x:.2f}' for x in task['target']['position']]}")

        # Phase 5: Execute
        print(f"\n[5/5] Executing pick & place...")
        from src.control.primitives import RobotPrimitives
        prims = RobotPrimitives(scene.robot, scene.scene)
        prims.open_gripper()
        scene.capture_frame()
        prims.pick_and_place(task['object']['position'], task['target']['position'])
        scene.capture_frame()

    # Save outputs
    print("\nSaving outputs...")
    scene.save_video("demo/output/vlm_grounding_demo.mp4", fps=30)
    with open("demo/output/vlm_grounding.json", "w") as f:
        json.dump(task, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  VLM Grounding Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("demo/output", exist_ok=True)
    main()
