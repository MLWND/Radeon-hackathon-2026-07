#!/usr/bin/env python3
"""
RoboPilot Demo Recording — Tight execution + recording, no long idle renders.

Follows official Genesis patterns:
- control_dofs_position held until replaced
- suction_cup: plan_path → weld → lift → place → release → settle
- No retreat step (official pattern)

Outputs:
    demo/output/demo.mp4         — Complete pick+place recording
    demo/output/before.png       — Scene before
    demo/output/after.png        — Scene after
    demo/output/states.json      — Verification data
"""
import genesis as gs
import numpy as np
import json, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)


def main():
    print("=" * 60)
    print("  RoboPilot Demo — Pick & Place")
    print("=" * 60)

    # ═══ INIT ══════════════════════════════════════════════
    gs.init(backend=gs.amdgpu)
    from src.envs.grasp_env import GraspEnv
    env = GraspEnv(num_envs=0, ctrl_dt=0.01)
    from src.vision.camera import CameraWrapper
    cam = CameraWrapper(env.vis_cam)

    print(f"  Genesis {gs.__version__} | GPU: AMD Radeon (ROCm)")
    print(f"  Robot: Franka Panda, 9 DOF | Objects: {list(env.entities.keys())}")

    # ═══ VLM ═══════════════════════════════════════════════
    print("\n--- VLM Perception ---")
    from openai import OpenAI
    from PIL import Image
    import base64, io

    img_before = cam.render()
    cam.render_and_save("demo/output/before.png")

    vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=60)
    INSTRUCTION = "Pick the red cube and place it next to the blue cube"

    pil_img = Image.fromarray(img_before)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    t0 = time.time()
    response = vlm_client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": (
                f'Return JSON: {{"pick":"name","place_relative":"description"}}\n'
                f'Objects: {list(env.entities.keys())}\nInstruction: {INSTRUCTION}'
            )},
        ]}],
        max_tokens=64,
    )
    vlm_ms = (time.time() - t0) * 1000
    raw = response.choices[0].message.content

    pick_name = "red_cube"
    place_relative = "next to blue_cube"
    try:
        parsed = json.loads(raw[raw.rfind("{"):raw.rfind("}") + 1])
        pick_name = parsed.get("pick", "red_cube")
        place_relative = parsed.get("place_relative", "next to blue_cube")
    except Exception:
        pass

    print(f"  Qwen3-VL-8B: {vlm_ms:.0f}ms → pick={pick_name}, place='{place_relative}'")

    from src.vision.qwen3vl import resolve_place_position
    place_pos = resolve_place_position(place_relative, env.entities)

    # ═══ RECORDING — camera-level recording ══════════════════
    # Records every render() call. We render at key moments to capture
    # the full pick+place cycle without long idle periods.
    print("\n--- Recording ---")
    env.start_recording()

    # 1. PICK
    t0 = time.time()
    lifted = env.suction_pick(pick_name, camera=env.vis_cam)
    pick_s = (time.time() - t0)
    obj = env.entities[pick_name]
    z_after_pick = float(np.asarray(obj.get_pos().cpu().numpy()).flatten()[2])
    print(f"  Pick: {'LIFTED' if lifted else 'FAILED'} (z={z_after_pick:.3f}) | {pick_s:.1f}s")

    # 2. PLACE
    env.suction_place(pick_name, place_pos, camera=env.vis_cam)

    final = np.asarray(obj.get_pos().cpu().numpy()).flatten()[:3]
    err = np.sqrt((final[0] - place_pos[0]) ** 2 + (final[1] - place_pos[1]) ** 2)
    print(f"  Place: error={err*100:.1f}cm at ({final[0]:.3f}, {final[1]:.3f})")

    # Render final state (moderate frames, won't drift arm)
    for _ in range(40):
        env.vis_cam.render()
        env.scene.step()
    env.stop_recording("demo/output/demo.mp4", fps=10)
    import os
    size_mb = os.path.getsize("demo/output/demo.mp4") / 1024 / 1024
    print(f"  Video saved: demo/output/demo.mp4 ({size_mb:.1f}MB)")

    # ═══ VERIFY ════════════════════════════════════════════
    print("\n--- Verification ---")
    img_after = cam.render()
    cam.render_and_save("demo/output/after.png")

    after_state = {}
    for name, e in env.entities.items():
        p = np.asarray(e.get_pos().cpu().numpy()).flatten()[:3]
        after_state[name] = [round(float(v), 4) for v in p]
        print(f"  {name}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # Save verification data
    states = {
        "task": INSTRUCTION,
        "vlm": {"model": "Qwen3-VL-8B", "pick": pick_name,
                "place_relative": place_relative, "inference_ms": round(vlm_ms)},
        "pick": {"object": pick_name, "lifted": bool(lifted),
                 "z_after": round(z_after_pick, 4), "duration_s": round(pick_s, 1)},
        "place": {"target": [round(float(v), 4) for v in place_pos],
                  "actual": [round(float(v), 4) for v in final],
                  "error_cm": round(float(err) * 100, 2)},
        "system": {"genesis": "1.2.2", "gpu": "AMD Radeon",
                   "backend": "rocm", "solver": "Newton"},
    }
    with open("demo/output/states.json", "w") as f:
        json.dump(states, f, indent=2)

    success = lifted and err < 0.10
    print(f"\n{'='*60}")
    print(f"  RESULT: {'SUCCESS' if success else 'NEEDS WORK'}")
    print(f"  Pick={lifted}, Place={err*100:.1f}cm")
    print(f"  Video: demo/output/demo.mp4")
    print(f"  Data:  demo/output/states.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
