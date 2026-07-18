#!/usr/bin/env python3
"""
RoboPilot Training — 100 Episodes

Runs the full pipeline 100 times, collecting statistics.
This is NOT RL training — it's a reliability benchmark.
"""
import genesis as gs
import numpy as np
import torch
import json, time, os, sys, logging
from PIL import Image
import base64, io

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("robopilot")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("demo/output", exist_ok=True)

N_EPISODES = 100


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_episode(scene, robot, ents, camera, vlm_client, prims, objects_def, T, S):
    """Run one episode. Returns result dict."""
    result = {"success": False, "error": 0.0, "clipped": False}

    try:
        # Reset
        obs = prims.reset()

        # VLM
        img = camera.render()[0]
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        t0 = time.time()
        response = vlm_client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": f'Return ONLY JSON: {{"pick":"name"}}\nObjects: {list(ents.keys())}'}
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
        except:
            pass

        target_obj = ents[pick_name]
        cube_pos = target_obj.get_pos().cpu().numpy()

        # PD Approach
        prims.pd_move_to_xyz([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15], steps=300)
        cube_now = target_obj.get_pos().cpu().numpy()
        prims.pd_move_to_xyz([cube_now[0], cube_now[1], cube_now[2] + 0.05], steps=200)

        # Weld
        grasped, force = prims.suction_grasp(pick_name)

        # PD Lift
        cube_now = target_obj.get_pos().cpu().numpy()
        prims.pd_move_to_xyz([cube_now[0], cube_now[1], cube_now[2] + 0.20], steps=300)
        cube_final = target_obj.get_pos().cpu().numpy()
        height_ok = cube_final[2] > cube_pos[2] + 0.03

        # PD Place
        goal = np.array([0.45, 0.0, T + S/2])
        prims.pd_move_to_xyz([goal[0], goal[1], goal[2] + 0.05], steps=300)
        prims.suction_release(pick_name)

        cube_placed = target_obj.get_pos().cpu().numpy()
        place_err = float(np.sqrt((cube_placed[0]-goal[0])**2 + (cube_placed[1]-goal[1])**2))

        # Retract
        prims.pd_move_to_xyz([0.5, 0.0, 0.3], steps=200)

        # Check clipping
        clipped = False
        for name in ents:
            if name != pick_name:
                p = ents[name].get_pos().cpu().numpy()
                orig = np.array(objects_def[name][:3])
                moved = np.sqrt((p[0]-orig[0])**2 + (p[1]-orig[1])**2)
                if moved > 0.02:
                    clipped = True

        result = {
            "success": grasped and height_ok and place_err < 0.10,
            "grasped": grasped,
            "height_ok": height_ok,
            "force": float(force),
            "error": place_err,
            "clipped": clipped,
            "vlm_ms": vlm_ms,
        }

    except Exception as e:
        result["error_msg"] = str(e)

    return result


def main():
    header(f"RoboPilot Training — {N_EPISODES} Episodes")

    from openai import OpenAI
    vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=30)

    # Build scene once
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.8), substeps=2),
        vis_options=gs.options.VisOptions(
            lights=[
                gs.options.vis.DirectionalLight(dir=(-1,-1,-1), color=(1,1,1), intensity=5),
                gs.options.vis.PointLight(pos=(2,-1,3), color=(1,0.95,0.9), intensity=8),
            ],
            ambient_light=(0.5, 0.5, 0.5),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.25, 0.25, 0.25)))
    scene.add_entity(gs.morphs.Box(size=(0.8, 1.0, 0.05), pos=(0.5, 0, 0.025)),
                     material=gs.materials.Kinematic(), surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55)))
    robot = scene.add_entity(gs.morphs.MJCF(
        file=os.path.join("venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml")))

    T, S = 0.05, 0.04
    objects_def = {
        "red_cube":    (0.60,  0.00, T+S/2, (0.9, 0.15, 0.15)),
        "blue_cube":   (0.55,  0.20, T+S/2, (0.15, 0.3,  0.9)),
        "green_cube":  (0.70, -0.10, T+S/2, (0.15, 0.75, 0.2)),
        "yellow_cube": (0.65,  0.15, T+S/2, (0.95, 0.85, 0.1)),
        "white_cube":  (0.50, -0.15, T+S/2, (0.9,  0.9,  0.9)),
        "orange_cube": (0.75,  0.05, T+S/2, (0.9,  0.5,  0.1)),
        "purple_cube": (0.60, -0.20, T+S/2, (0.6,  0.2,  0.8)),
        "cyan_cube":   (0.70,  0.20, T+S/2, (0.1,  0.7,  0.8)),
    }
    ents = {}
    for name, (x, y, z, c) in objects_def.items():
        ents[name] = scene.add_entity(
            gs.morphs.Box(size=(S, S, S), pos=(x, y, z)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Smooth(color=c, roughness=0.3))

    scene.add_entity(gs.morphs.Box(size=(0.15, 0.15, 0.002), pos=(0.45, 0, T+0.001)),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Smooth(color=(0.2, 0.5, 1.0, 0.4), roughness=0.1))

    camera = scene.add_camera(res=(640, 480), pos=(1.5, -2.0, 1.6),
                              lookat=(0.5, 0, 0.08), fov=45)
    scene.build()
    scene.step(200)

    from src.control.primitives import RobotPrimitives
    prims = RobotPrimitives(robot, scene, ents)

    # Run episodes
    results = []
    t_start = time.time()

    for ep in range(N_EPISODES):
        t0 = time.time()
        r = run_episode(scene, robot, ents, camera, vlm_client, prims, objects_def, T, S)
        r["episode"] = ep + 1
        r["wall_time"] = time.time() - t0
        results.append(r)

        # Progress
        successes = sum(1 for r in results if r["success"])
        clipped = sum(1 for r in results if r.get("clipped"))
        avg_err = np.mean([r["error"] for r in results])
        avg_vlm = np.mean([r.get("vlm_ms", 0) for r in results])

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{ep+1:3d}/{N_EPISODES}] success={successes}/{ep+1} "
                  f"({successes/(ep+1)*100:.0f}%) clip={clipped} "
                  f"avg_err={avg_err*100:.1f}cm avg_vlm={avg_vlm:.0f}ms")

    # Final summary
    total_time = time.time() - t_start
    successes = sum(1 for r in results if r["success"])
    clipped = sum(1 for r in results if r.get("clipped"))
    avg_err = np.mean([r["error"] for r in results])
    avg_vlm = np.mean([r.get("vlm_ms", 0) for r in results])
    avg_force = np.mean([r.get("force", 0) for r in results])
    grasped = sum(1 for r in results if r.get("grasped"))

    print(f"\n{'='*60}")
    print(f"  Training Results — {N_EPISODES} Episodes")
    print(f"{'='*60}")
    print(f"  Success:     {successes}/{N_EPISODES} ({successes/N_EPISODES*100:.1f}%)")
    print(f"  Grasped:     {grasped}/{N_EPISODES} ({grasped/N_EPISODES*100:.1f}%)")
    print(f"  Clipped:     {clipped}/{N_EPISODES} ({clipped/N_EPISODES*100:.1f}%)")
    print(f"  Avg error:   {avg_err*100:.1f}cm")
    print(f"  Avg force:   {avg_force:.2f}N")
    print(f"  Avg VLM:     {avg_vlm:.0f}ms")
    print(f"  Total time:  {total_time:.1f}s ({total_time/N_EPISODES:.1f}s/ep)")
    print(f"{'='*60}")

    # Save results
    clean_results = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                cr[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                cr[k] = bool(v)
            else:
                cr[k] = v
        clean_results.append(cr)

    with open("demo/output/training_results.json", "w") as f:
        json.dump({
            "n_episodes": N_EPISODES,
            "success_rate": successes / N_EPISODES,
            "grasp_rate": grasped / N_EPISODES,
            "clip_rate": clipped / N_EPISODES,
            "avg_error_cm": float(avg_err * 100),
            "avg_force_n": float(avg_force),
            "avg_vlm_ms": float(avg_vlm),
            "total_time_s": total_time,
            "episodes": clean_results,
        }, f, indent=2)
    print(f"  Saved: demo/output/training_results.json")


if __name__ == "__main__":
    main()
