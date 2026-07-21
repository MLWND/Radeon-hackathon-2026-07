"""
Residual RL Environment — RL adjusts placement target, scripted controller executes.

Architecture (per episode):
    1. RL policy observes scene → outputs (dx, dy) target offset
    2. Adjusted target = base_target + offset
    3. Scripted controller executes pick → place to adjusted target
    4. Reward = exp(-placement_error) — closer to 0 = better

This is "Residual RL" in task space: RL corrects WHERE to place,
not HOW to move the arm. The scripted controller handles all motion.
"""
import math
from typing import Dict

import numpy as np
import torch
import genesis as gs

from src.envs.grasp_env import GraspEnv


class ResidualGraspEnv:
    """RL fine-tunes scripted controller placement target.

    Action space: 2D [dx, dy] offset in meters (clamped to ±3cm)
    Observation:  14D (finger_pos-obj_pos, finger_quat, obj_pos, obj_quat)
    Reward:       exp(-placement_error) × 10 — scaled for learning signal

    One episode = one RL action = one scripted pick+place cycle.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = None,
        ctrl_dt: float = 0.01,
        target_pos: list = None,
    ):
        self.num_envs = num_envs
        self.num_actions = 2  # [dx, dy] target offset
        self.device = device or str(gs.device)
        self.ctrl_dt = ctrl_dt
        self.target_pos = target_pos or [0.4, 0.2, 0.02]
        self.max_episode_length = 1  # One RL action per episode

        # Create base GraspEnv for scripted controller
        self.env = GraspEnv(
            num_envs=1,  # Scripted always single-env
            device=device,
            ctrl_dt=ctrl_dt,
            action_scale=0.05,
        )

        # RL buffers
        B = max(num_envs, 1)
        self.episode_length_buf = torch.zeros(B, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(B, device=self.device, dtype=torch.bool)
        self._base_error = None

        # Custom PPO storage params
        self._num_obs = 14

        self.reset()

    def reset(self, envs_idx=None):
        """Reset env: run base scripted pick+place, return observation."""
        self.env.reset()
        self._run_scripted(self.target_pos)
        self._base_error = self._measure_error()
        self.episode_length_buf.zero_()
        self.reset_buf.fill_(False)
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        """Apply target offset, execute scripted place, return (obs, reward, done, info).

        actions: [B, 2] — (dx, dy) offset in meters, clamped to ±3cm
        """
        offset_x = float(actions[..., 0].clamp(-0.03, 0.03).mean())
        offset_y = float(actions[..., 1].clamp(-0.03, 0.03).mean())

        # Adjust target position
        adjusted = [
            self.target_pos[0] + offset_x,
            self.target_pos[1] + offset_y,
            self.target_pos[2],
        ]

        # Execute scripted pick+place to adjusted target
        pick_name = "red_cube"
        if pick_name not in self.env.entities:
            pick_name = list(self.env.entities.keys())[0]

        self.env.suction_pick(pick_name)
        self.env.suction_place(pick_name, adjusted)

        # Reward: negative error (scaled for learning signal)
        error = self._measure_error()
        reward = -error * 100.0  # tensor, ~-3.0 for 3cm, ~0 for 0cm

        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        self.reset_buf[:] = done

        info = {
            "base_error": 0.0,
            "current_error": float(error),
            "improvement": float(self._base_error - error),
        }

        return self.get_observations(), reward, self.reset_buf, info

    def get_observations(self) -> "TensorDict":
        """14-dim observation, broadcast to num_envs."""
        from tensordict import TensorDict
        B = max(self.num_envs, 1)
        ee = self.env.robot.ee_link
        obj = self.env.entities[self.env._target_obj]
        obs = torch.cat([
            ee.get_pos().squeeze(0) - obj.get_pos().squeeze(0),
            ee.get_quat().squeeze(0),
            obj.get_pos().squeeze(0),
            obj.get_quat().squeeze(0),
        ], dim=-1)  # [14]
        obs = obs.unsqueeze(0).expand(B, -1).contiguous()  # [B, 14]
        return TensorDict({"policy": obs}, batch_size=[obs.shape[0]])

    def _run_scripted(self, pos):
        """Run scripted pick and place to given position."""
        pick_name = "red_cube"
        if pick_name not in self.env.entities:
            pick_name = list(self.env.entities.keys())[0]
        self.env.suction_pick(pick_name)
        self.env.suction_place(pick_name, pos)

    def _measure_error(self) -> torch.Tensor:
        """XY distance from placed object to target."""
        obj = self.env.entities.get("red_cube") or list(self.env.entities.values())[0]
        pos = obj.get_pos().squeeze(0)[:2]
        target = torch.tensor(self.target_pos[:2], dtype=torch.float32, device=self.device)
        return torch.norm(pos - target, dim=-1)

    @property
    def cfg(self):
        return {
            "num_envs": self.num_envs,
            "num_actions": 2,
            "ctrl_dt": self.ctrl_dt,
            "target_pos": self.target_pos,
        }
