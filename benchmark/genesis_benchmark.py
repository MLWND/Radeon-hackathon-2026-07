#!/usr/bin/env python3
"""
Genesis CPU vs GPU Benchmark — for AMD ROCm evaluation.

Measures:
- Scene build time
- Physics simulation FPS
- Camera render time
- IK solve time

Usage:
    python benchmark/genesis_benchmark.py
"""
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def benchmark_backend(backend_name, backend):
    """Benchmark Genesis on a specific backend."""
    import genesis as gs
    import numpy as np

    print(f"\n{'='*50}")
    print(f"  Benchmarking: {backend_name}")
    print(f"{'='*50}")

    gs.init(backend=backend)

    # ── Scene build ──────────────────────────────────────
    t0 = time.time()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    for i, (name, color, pos) in enumerate([
        ("red", (1, 0, 0), (0.65, 0.0, 0.02)),
        ("blue", (0, 1, 0), (0.4, 0.2, 0.02)),
        ("green", (0, 0, 1), (0.7, -0.1, 0.02)),
    ]):
        scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos),
            surface=gs.surfaces.Plastic(color=color),
        )
    camera = scene.add_camera(res=(640, 480), pos=(1.5, -2.0, 1.6), lookat=(0.5, 0, 0.0), fov=45)
    scene.build()
    build_time = time.time() - t0
    print(f"  Scene build: {build_time:.2f}s")

    # Set PD gains
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

    # Warm up
    scene.step(100)

    # ── Physics FPS ──────────────────────────────────────
    n_steps = 500
    t0 = time.time()
    for _ in range(n_steps):
        scene.step()
    physics_time = time.time() - t0
    physics_fps = n_steps / physics_time
    print(f"  Physics FPS: {physics_fps:.1f} ({n_steps} steps in {physics_time:.2f}s)")

    # ── Camera render ────────────────────────────────────
    n_renders = 50
    t0 = time.time()
    for _ in range(n_renders):
        camera.render()
    render_time = (time.time() - t0) / n_renders * 1000
    print(f"  Camera render: {render_time:.1f}ms per frame ({n_renders} frames)")

    # ── IK solve ─────────────────────────────────────────
    ee = robot.get_link("hand")
    n_ik = 100
    t0 = time.time()
    for _ in range(n_ik):
        robot.inverse_kinematics(
            link=ee,
            pos=np.array([0.65, 0.0, 0.25]),
            quat=np.array([0, 1, 0, 0]),
        )
    ik_time = (time.time() - t0) / n_ik * 1000
    print(f"  IK solve: {ik_time:.2f}ms per call ({n_ik} calls)")

    return {
        "backend": backend_name,
        "build_time": build_time,
        "physics_fps": physics_fps,
        "render_ms": render_time,
        "ik_ms": ik_time,
    }


def main():
    import genesis as gs
    results = []

    # Benchmark GPU
    try:
        r = benchmark_backend("AMD GPU (ROCm)", gs.amdgpu)
        results.append(r)
    except Exception as e:
        print(f"  GPU benchmark failed: {e}")

    # Benchmark CPU
    try:
        r = benchmark_backend("CPU", gs.cpu)
        results.append(r)
    except Exception as e:
        print(f"  CPU benchmark failed: {e}")

    # ── Summary table ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Backend':<20} {'Build':>8} {'FPS':>8} {'Render':>10} {'IK':>10}")
    print(f"{'':─<20} {'':─>8} {'':─>8} {'':─>10} {'':─>10}")
    for r in results:
        print(f"{r['backend']:<20} {r['build_time']:>7.2f}s {r['physics_fps']:>7.1f} "
              f"{r['render_ms']:>8.1f}ms {r['ik_ms']:>8.2f}ms")

    if len(results) == 2:
        gpu, cpu = results[0], results[1]
        print(f"\n  Speedup (GPU vs CPU):")
        print(f"    Physics: {cpu['physics_fps']/gpu['physics_fps']:.1f}x faster" if gpu['physics_fps'] > 0 else "")
        print(f"    Render:  {cpu['render_ms']/gpu['render_ms']:.1f}x faster" if gpu['render_ms'] > 0 else "")
        print(f"    IK:      {cpu['ik_ms']/gpu['ik_ms']:.1f}x faster" if gpu['ik_ms'] > 0 else "")


if __name__ == "__main__":
    main()
