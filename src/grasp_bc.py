#!/usr/bin/env python3
"""
Behavior Cloning — distill RL teacher to vision-based student.

Stage 2 of the official Genesis manipulation training pipeline.
Student policy takes stereo RGB images → predicts 6D delta-EE actions.

Usage:
    python src/grasp_bc.py --exp_name grasp --max_iterations 200
"""
import argparse
import os
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

import genesis as gs
from src.envs.grasp_env import GraspEnv


# ── Policy (CNN vision encoder + action head) ──────────────────

class VisionEncoder(nn.Module):
    """Simple CNN for stereo RGB images."""

    def __init__(self, in_channels=6, img_size=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


class StudentPolicy(nn.Module):
    """Vision-based policy: stereo RGB → 6D delta-EE action."""

    def __init__(self, action_dim=6, state_dim=7, img_size=64):
        super().__init__()
        self.vision = VisionEncoder(in_channels=6, img_size=img_size)
        self.action_head = nn.Sequential(
            nn.Linear(128 + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, rgb_obs, state_obs):
        """rgb_obs: (B, 6, H, W), state_obs: (B, 7) ee_pose."""
        vis_feat = self.vision(rgb_obs)
        x = torch.cat([vis_feat, state_obs], dim=-1)
        return self.action_head(x)


# ── Experience Buffer ──────────────────────────────────────────

class ExperienceBuffer:
    def __init__(self, num_envs, max_size, img_shape, state_dim, action_dim, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.rgb = torch.zeros((max_size, *img_shape), device=device)
        self.state = torch.zeros((max_size, state_dim), device=device)
        self.actions = torch.zeros((max_size, action_dim), device=device)

    def add(self, rgb, state, actions):
        n = rgb.shape[0]
        end = min(self.ptr + n, self.max_size)
        actual = end - self.ptr
        self.rgb[self.ptr:end] = rgb[:actual]
        self.state[self.ptr:end] = state[:actual]
        self.actions[self.ptr:end] = actions[:actual]
        self.ptr = end % self.max_size
        self.size = min(self.size + n, self.max_size)

    def get_batches(self, num_batches, num_epochs):
        for _ in range(num_epochs):
            idx = torch.randperm(self.size, device=self.device)
            batch_size = self.size // num_batches
            for i in range(num_batches):
                b = idx[i * batch_size:(i + 1) * batch_size]
                yield {
                    "rgb_obs": self.rgb[b],
                    "state_obs": self.state[b],
                    "actions": self.actions[b],
                }

    def clear(self):
        self.ptr = 0
        self.size = 0


# ── Behavior Cloning Trainer ───────────────────────────────────

class BehaviorCloning:
    def __init__(self, env, cfg, teacher_policy, device="cpu"):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.teacher = teacher_policy
        self.num_steps = cfg["num_steps_per_env"]

        img_shape = (6, env.image_size[1], env.image_size[0])
        self.policy = StudentPolicy(
            action_dim=6, state_dim=7, img_size=env.image_size[1],
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg["learning_rate"])

        self.buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=cfg["buffer_size"],
            img_shape=img_shape,
            state_dim=7,
            action_dim=6,
            device=device,
        )

    def learn(self, num_learning_iterations, log_dir):
        tf_writer = SummaryWriter(log_dir)
        self.buffer.clear()

        for it in range(num_learning_iterations):
            # Collect data from RL teacher
            t0 = time.time()
            self._collect()
            fwd_time = time.time() - t0

            # Train student
            t0 = time.time()
            total_loss = 0.0
            n_batches = 0
            for batch in self.buffer.get_batches(
                    self.cfg.get("num_mini_batches", 4),
                    self.cfg["num_epochs"]):
                pred = self.policy(batch["rgb_obs"], batch["state_obs"])
                loss = F.mse_loss(pred, batch["actions"])
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg["max_grad_norm"])
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            bwd_time = time.time() - t0

            avg_loss = total_loss / max(n_batches, 1)
            fps = (self.num_steps * self.env.num_envs) / max(fwd_time, 0.01)

            if (it + 1) % self.cfg["log_freq"] == 0:
                tf_writer.add_scalar("loss/action", avg_loss, it)
                tf_writer.add_scalar("speed/fps", fps, it)
                print(f"  Iter {it+1:04d} | Loss: {avg_loss:.6f} | "
                      f"FPS: {int(fps)} | Buf: {self.buffer.size}")

            if (it + 1) % self.cfg["save_freq"] == 0:
                self.save(os.path.join(log_dir, f"bc_{it+1:04d}.pt"))

        tf_writer.close()

    def _collect(self):
        """Collect experience using RL teacher."""
        obs = self.env.get_observations()
        with torch.inference_mode():
            for _ in range(self.num_steps):
                rgb = self.env.get_stereo_rgb(normalize=True)
                if rgb is None:
                    # Fallback: use dummy stereo images
                    B = self.env.num_envs or 1
                    rgb = torch.zeros(
                        B, 6, self.env.image_size[1], self.env.image_size[0],
                        device=self.device)
                state = self.env.robot.ee_pose  # (B, 7)
                teacher_action = self.teacher(obs).detach()

                self.buffer.add(rgb, state, teacher_action)
                obs, rew, done, info = self.env.step(teacher_action)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True))


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-B", "--num_envs", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=200)
    args = parser.parse_args()

    gs.init(backend=gs.amdgpu, logging_level="warning")

    log_dir = Path("logs") / f"{args.exp_name}_rl"
    assert log_dir.exists(), f"RL checkpoint dir not found: {log_dir}"

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, rl_cfg = pickle.load(f)

    env_cfg["num_envs"] = args.num_envs
    env = GraspEnv(
        num_envs=env_cfg["num_envs"],
        ctrl_dt=env_cfg["ctrl_dt"],
        episode_length_s=env_cfg["episode_length_s"],
        action_scale=env_cfg["action_scale"],
    )

    # Load RL teacher
    runner = OnPolicyRunner(env, rl_cfg, log_dir, device=gs.device)
    ckpts = sorted(log_dir.glob("model_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    assert ckpts, f"No RL checkpoints in {log_dir}"
    runner.load(ckpts[-1])
    teacher = runner.get_inference_policy(device=gs.device)
    print(f"Loaded teacher from {ckpts[-1]}")

    # Train BC student
    bc_cfg = {
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        "buffer_size": 1000,
        "log_freq": 10,
        "save_freq": 50,
    }
    bc_dir = Path("logs") / f"{args.exp_name}_bc"
    bc_dir.mkdir(parents=True, exist_ok=True)

    bc = BehaviorCloning(env, bc_cfg, teacher, device=gs.device)
    bc.learn(args.max_iterations, bc_dir)


if __name__ == "__main__":
    main()
