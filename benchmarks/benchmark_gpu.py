#!/usr/bin/env python3
"""
Benchmark: Genesis CPU vs AMD GPU (ROCm) performance comparison.

Measures:
- Scene build time
- Physics simulation FPS
- Camera render time
- IK solve time
- Full pick-and-place pipeline time

Usage:
    python benchmarks/benchmark_gpu.py
    python benchmarks/benchmark_gpu.py --cpu-only
    python benchmarks/benchmark_gpu.py --gpu-only
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch


def benchmark_physics(backend, num_steps=1000, num_envs=1):
    """Benchmark physics simulation FPS."""
    import genesis as gs

    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.5, 0, 0.02)),
        surface=gs.surfaces.Smooth(color=(1, 0, 0)),
    )
    scene.build(n_envs=num_envs)
    scene.step(50)

    # Warm up
    for _ in range(10):
        scene.step()

    # Benchmark
    t0 = time.time()
    for _ in range(num_steps):
        scene.step()
    elapsed = time.time() - t0
    fps = num_steps / elapsed

    return {
        "backend": str(backend),
        "num_envs": num_envs,
        "num_steps": num_steps,
        "elapsed_s": round(elapsed, 2),
        "fps": round(fps, 1),
    }


def benchmark_ik(backend, num_solves=100):
    """Benchmark IK solve time."""
    import genesis as gs

    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    scene.step(10)

    ee = franka.get_link("hand")
    targets = [
        torch.tensor([0.5, 0.0, 0.3], dtype=torch.float32),
        torch.tensor([0.4, 0.1, 0.2], dtype=torch.float32),
        torch.tensor([0.6, -0.1, 0.25], dtype=torch.float32),
    ]

    # Warm up
    for t in targets:
        franka.inverse_kinematics(link=ee, pos=t)

    # Benchmark
    t0 = time.time()
    for i in range(num_solves):
        t = targets[i % len(targets)]
        franka.inverse_kinematics(link=ee, pos=t)
    elapsed = time.time() - t0

    return {
        "backend": str(backend),
        "num_solves": num_solves,
        "elapsed_s": round(elapsed, 3),
        "per_solve_ms": round(elapsed / num_solves * 1000, 2),
    }


def benchmark_camera(backend, num_renders=50):
    """Benchmark camera render time."""
    import genesis as gs

    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(res=(640, 480), pos=(1.5, -1, 1.2), lookat=(0.5, 0, 0), fov=50)
    scene.build()
    scene.step(50)

    # Warm up
    cam.render()

    # Benchmark
    t0 = time.time()
    for _ in range(num_renders):
        cam.render()
    elapsed = time.time() - t0

    return {
        "backend": str(backend),
        "resolution": "640x480",
        "num_renders": num_renders,
        "elapsed_s": round(elapsed, 3),
        "per_render_ms": round(elapsed / num_renders * 1000, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="benchmarks/results.json")
    args = parser.parse_args()

    results = {}

    if not args.cpu_only:
        print("\n=== GPU (AMD ROCm) Benchmarks ===")
        print("\nPhysics simulation...")
        results["gpu_physics_1env"] = benchmark_physics(gs.amdgpu, args.num_steps, 1)
        print(f"  FPS: {results['gpu_physics_1env']['fps']}")

        results["gpu_physics_4env"] = benchmark_physics(gs.amdgpu, args.num_steps // 4, 4)
        print(f"  FPS (4 envs): {results['gpu_physics_4env']['fps']}")

        print("\nIK solving...")
        results["gpu_ik"] = benchmark_ik(gs.amdgpu)
        print(f"  Per solve: {results['gpu_ik']['per_solve_ms']}ms")

        print("\nCamera rendering...")
        results["gpu_camera"] = benchmark_camera(gs.amdgpu)
        print(f"  Per render: {results['gpu_camera']['per_render_ms']}ms")

    if not args.gpu_only:
        print("\n=== CPU Benchmarks ===")
        print("\nPhysics simulation...")
        results["cpu_physics"] = benchmark_physics(gs.cpu, args.num_steps, 1)
        print(f"  FPS: {results['cpu_physics']['fps']}")

        print("\nIK solving...")
        results["cpu_ik"] = benchmark_ik(gs.cpu)
        print(f"  Per solve: {results['cpu_ik']['per_solve_ms']}ms")

        print("\nCamera rendering...")
        results["cpu_camera"] = benchmark_camera(gs.cpu)
        print(f"  Per render: {results['cpu_camera']['per_render_ms']}ms")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print comparison table
    if "gpu_physics_1env" in results and "cpu_physics" in results:
        print("\n=== CPU vs GPU Comparison ===")
        print(f"{'Metric':<25} {'CPU':>10} {'GPU':>10} {'Speedup':>10}")
        print("-" * 55)
        gpu_fps = results["gpu_physics_1env"]["fps"]
        cpu_fps = results["cpu_physics"]["fps"]
        print(f"{'Physics FPS':<25} {cpu_fps:>10} {gpu_fps:>10} {gpu_fps/cpu_fps:>9.1f}x")

        gpu_ik = results["gpu_ik"]["per_solve_ms"]
        cpu_ik = results["cpu_ik"]["per_solve_ms"]
        print(f"{'IK solve (ms)':<25} {cpu_ik:>10.2f} {gpu_ik:>10.2f} {cpu_ik/gpu_ik:>9.1f}x")

        gpu_cam = results["gpu_camera"]["per_render_ms"]
        cpu_cam = results["cpu_camera"]["per_render_ms"]
        print(f"{'Camera render (ms)':<25} {cpu_cam:>10.2f} {gpu_cam:>10.2f} {cpu_cam/gpu_cam:>9.1f}x")


if __name__ == "__main__":
    main()
