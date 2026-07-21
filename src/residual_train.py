#!/usr/bin/env python3
"""
Residual RL Training — RL learns target offset to improve placement accuracy.

Usage:
    python src/residual_train.py --max_iterations 200
    python src/residual_train.py -B 64 --max_iterations 300
"""
import argparse
from pathlib import Path
import pickle

from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="residual_v2")
    parser.add_argument("-B", "--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gs.init(
        backend=gs.amdgpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    from src.envs.residual_env import ResidualGraspEnv

    env_cfg = {"num_envs": args.num_envs, "ctrl_dt": 0.01}

    rl_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 64, 32],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 64, 32],
            "activation": "relu",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 1,  # One RL action per episode
        "save_interval": 50,
        "run_name": args.exp_name,
        "logger": "tensorboard",
    }

    log_dir = Path(f"logs/{args.exp_name}_residual_rl")
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, rl_cfg), f)

    print(f"Creating ResidualGraspEnv ({args.num_envs} envs, 2D target offset)...")
    env = ResidualGraspEnv(num_envs=args.num_envs, ctrl_dt=env_cfg["ctrl_dt"])
    print(f"  num_actions: {env.num_actions} (dx, dy target offset)")
    print(f"  base_error: {env._base_error}")

    print(f"Starting training ({args.max_iterations} iterations)...")
    runner = OnPolicyRunner(env, rl_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    print("Residual RL training complete!")


if __name__ == "__main__":
    main()
