#!/usr/bin/env python3
"""
GraspEnv PPO Training — follows official Genesis grasp_train.py pattern.

Usage:
    python src/train_grasp.py --stage=rl                   # Train RL teacher
    python src/train_grasp.py --stage=rl -B 64             # 64 parallel envs
    python src/train_grasp.py --stage=rl --max_iterations 300
"""
import argparse
import pickle
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from src.envs.grasp_env import GraspEnv


def get_rl_cfg():
    return {
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
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "logger": "tensorboard",
    }


def get_env_cfg():
    return {
        "num_envs": 2048,
        "ctrl_dt": 0.01,
        "episode_length_s": 3.0,
        "action_scale": 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    gs.init(
        backend=gs.amdgpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    env_cfg = get_env_cfg()
    env_cfg["num_envs"] = args.num_envs
    rl_cfg = get_rl_cfg()
    rl_cfg["run_name"] = args.exp_name

    log_dir = Path("logs") / f"{args.exp_name}_rl"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, rl_cfg), f)

    env = GraspEnv(
        num_envs=env_cfg["num_envs"],
        ctrl_dt=env_cfg["ctrl_dt"],
        episode_length_s=env_cfg["episode_length_s"],
        action_scale=env_cfg["action_scale"],
    )

    runner = OnPolicyRunner(env, rl_cfg, log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
