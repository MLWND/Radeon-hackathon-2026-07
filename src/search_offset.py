#!/usr/bin/env python3
"""
Grid/Random Search for Optimal Placement Target Offset.

Since the residual RL problem is only 2D (dx, dy offset to placement target),
we can use simple search instead of PPO. Much faster and more reliable.

Usage:
    python src/search_offset.py
    python src/search_offset.py --method grid --n-trials 5
    python src/search_offset.py --method random --n-samples 100
"""
import argparse
import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.makedirs("benchmark/output", exist_ok=True)


def evaluate_offset(env, dx, dy, n_trials=1):
    """Evaluate a single offset, return mean error."""
    errors = []
    for _ in range(n_trials):
        env.reset()
        # Manually run scripted sequence with adjusted target
        pick_name = "red_cube"
        if pick_name not in env.env.entities:
            pick_name = list(env.env.entities.keys())[0]
        env.env.suction_pick(pick_name)
        adjust = [
            env.target_pos[0] + dx,
            env.target_pos[1] + dy,
            env.target_pos[2],
        ]
        env.env.suction_place(pick_name, adjust)
        errors.append(float(env._measure_error()))
    return sum(errors) / len(errors)


def grid_search(env, n_trials=3, step=0.005):
    """Grid search over [-3cm, +3cm]."""
    best_error = float("inf")
    best_offset = (0.0, 0.0)
    results = []

    offsets = [round(x * step, 3) for x in range(-6, 7)]  # -0.03 to +0.03
    total = len(offsets) ** 2
    count = 0

    print(f"  Grid search: {len(offsets)}×{len(offsets)} = {total} points")
    t0 = time.time()

    for dx in offsets:
        for dy in offsets:
            err = evaluate_offset(env, dx, dy, n_trials)
            results.append({"dx": dx, "dy": dy, "error_cm": round(err * 100, 2)})
            count += 1
            if err < best_error:
                best_error = err
                best_offset = (dx, dy)
            if count % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"    {count}/{total} | best={best_error*100:.2f}cm @ ({best_offset[0]:+.3f}, {best_offset[1]:+.3f}) | ETA {eta:.0f}s")

    return best_offset, best_error, results


def random_search(env, n_samples=100, n_trials=3):
    """Random search over [-3cm, +3cm]."""
    import random
    import numpy as np

    best_error = float("inf")
    best_offset = (0.0, 0.0)
    results = []

    print(f"  Random search: {n_samples} samples")

    for i in range(n_samples):
        dx = np.random.uniform(-0.03, 0.03)
        dy = np.random.uniform(-0.03, 0.03)
        err = evaluate_offset(env, dx, dy, n_trials)
        results.append({"dx": dx, "dy": dy, "error_cm": round(err * 100, 2)})
        if err < best_error:
            best_error = err
            best_offset = (dx, dy)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_samples} | best={best_error*100:.2f}cm @ ({best_offset[0]:+.3f}, {best_offset[1]:+.3f})")

    return best_offset, best_error, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["grid", "random"], default="random")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--step", type=float, default=0.005)
    args = parser.parse_args()

    import genesis as gs
    gs.init(backend=gs.amdgpu, logging_level="warning")

    from src.envs.residual_env import ResidualGraspEnv
    env = ResidualGraspEnv(num_envs=1, ctrl_dt=0.01)

    # Baseline
    base_err = evaluate_offset(env, 0, 0, args.n_trials)
    print(f"\n{'='*50}")
    print(f"  Baseline (no offset): {base_err*100:.2f}cm")
    print(f"{'='*50}")

    t0 = time.time()

    if args.method == "grid":
        best_offset, best_error, results = grid_search(env, args.n_trials, args.step)
    else:
        best_offset, best_error, results = random_search(env, args.n_samples, args.n_trials)

    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Baseline:     {base_err*100:.2f}cm")
    print(f"  Best offset:  ({best_offset[0]:+.4f}, {best_offset[1]:+.4f})")
    print(f"  Best error:   {best_error*100:.2f}cm")
    print(f"  Improvement:  {(base_err - best_error)*100:.2f}cm")
    print(f"  Time:         {elapsed:.0f}s ({len(results)} evals)")

    # Save results
    with open("benchmark/output/offset_search.json", "w") as f:
        json.dump({
            "method": args.method,
            "baseline_cm": round(base_err * 100, 2),
            "best_offset": list(best_offset),
            "best_error_cm": round(best_error * 100, 2),
            "improvement_cm": round((base_err - best_error) * 100, 2),
            "results": sorted(results, key=lambda r: r["error_cm"])[:10],  # Top 10
            "n_evals": len(results),
            "time_s": round(elapsed),
        }, f, indent=2)

    print(f"\n  Top 10 results saved to benchmark/output/offset_search.json")


if __name__ == "__main__":
    main()
