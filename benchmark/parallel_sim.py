"""
Parallel Simulation Benchmark — AMD GPU Showcase.
Genesis supports parallel environments via n_envs.

Tests: 1, 4, 8, 16, 32, 64 environments
Measures: FPS, throughput, GPU memory
"""
import genesis as gs
import time
import json
import os


def build_parallel_scene(n_envs: int, env_spacing: float = 2.0):
    """Build a parallel tabletop scene with n_envs environments."""
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.8)),
        show_viewer=False,
    )

    # Ground
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())

    # Add n_envs robots
    robots = []
    for i in range(n_envs):
        row = i // int(n_envs**0.5 + 1)
        col = i % int(n_envs**0.5 + 1)
        x = col * env_spacing
        y = row * env_spacing
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file="/opt/venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml",
                pos=(x, y, 0.75),
            ),
        )
        robots.append(robot)

    # Add objects per environment
    objects = []
    for i in range(n_envs):
        row = i // int(n_envs**0.5 + 1)
        col = i % int(n_envs**0.5 + 1)
        x = col * env_spacing + 0.3
        y = row * env_spacing + 0.15
        obj = scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(x, y, 0.42)),
            material=gs.materials.Rigid(),
        )
        objects.append(obj)

    scene.build()
    return scene, robots, objects


def benchmark_parallel(n_envs_list=None, steps=200):
    """Run parallel benchmark across different environment counts."""
    if n_envs_list is None:
        n_envs_list = [1, 4, 8, 16, 32]

    results = []

    for n_envs in n_envs_list:
        print(f"\n{'='*50}")
        print(f"  Testing {n_envs} parallel environments")
        print(f"{'='*50}")

        try:
            # Build scene
            start = time.time()
            scene, robots, objects = build_parallel_scene(n_envs)
            build_time = time.time() - start

            # Warmup
            for _ in range(20):
                scene.step()

            # Benchmark
            start = time.time()
            for _ in range(steps):
                scene.step()
            elapsed = time.time() - start

            fps = steps / elapsed
            throughput = fps * n_envs  # Total environment steps per second

            # Get GPU memory
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
            else:
                gpu_mem = 0

            result = {
                "n_envs": n_envs,
                "fps": round(fps, 1),
                "throughput": round(throughput, 1),
                "build_time_s": round(build_time, 2),
                "total_time_s": round(elapsed, 2),
                "gpu_memory_gb": round(gpu_mem, 2),
                "steps": steps,
            }
            results.append(result)

            print(f"  FPS: {fps:.1f}")
            print(f"  Throughput: {throughput:.1f} env-steps/s")
            print(f"  Build time: {build_time:.1f}s")
            print(f"  GPU memory: {gpu_mem:.1f} GB")

            # Cleanup
            del scene, robots, objects
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"n_envs": n_envs, "error": str(e)})

    return results


def print_results_table(results):
    """Print results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"  Parallel Simulation Benchmark — AMD Radeon PRO W7900D")
    print(f"{'='*70}")
    print(f"  {'Envs':<8} {'FPS':<10} {'Throughput':<15} {'Build (s)':<12} {'GPU (GB)':<10}")
    print(f"  {'-'*55}")
    for r in results:
        if "error" in r:
            print(f"  {r['n_envs']:<8} ERROR: {r['error']}")
        else:
            print(f"  {r['n_envs']:<8} {r['fps']:<10} {r['throughput']:<15} {r['build_time_s']:<12} {r['gpu_memory_gb']:<10}")
    print(f"{'='*70}")


def main():
    print("Parallel Simulation Benchmark")
    print("Testing Genesis GPU parallel environments on AMD Radeon PRO W7900D")

    results = benchmark_parallel([1, 4, 8, 16], steps=200)
    print_results_table(results)

    # Save results
    os.makedirs("benchmark/results", exist_ok=True)
    with open("benchmark/results/parallel_sim.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark/results/parallel_sim.json")


if __name__ == "__main__":
    main()
