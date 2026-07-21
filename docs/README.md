# RoboPilot

**Vision-Language Physical AI Robot on AMD Radeon GPU**

> Qwen3-VL-8B (Brain) + Genesis (Physics) + Suction Gripper (Hand) + ROCm (Compute)

## Quick Start

```bash
# 1. Setup (one-time)
bash setup.sh

# 2. Run demo
source venv/bin/activate
python3 demo/full_demo.py

# 3. Run E2E test (all modules)
python3 demo/test_e2e.py

# 4. Train RL policy
python3 src/train_grasp.py -B 64 --max_iterations 300

# 5. Behavior cloning (after RL training)
python3 src/grasp_bc.py --max_iterations 200
```

## Pipeline

```
User: "Pick the red cube and place it next to the blue cube"
  │
  ├─ [1.8s]   Qwen3-VL-8B via vLLM → object detection + spatial grounding
  ├─ [0.1s]   Task Planner → keyword-based instruction decomposition
  ├─ [0.1s]   Action Scheduler → sequence execution
  ├─ [6.4s]   GraspEnv.suction_pick → OMPL plan_path + weld + PD lift
  ├─ [1.0s]   GraspEnv.suction_place → PD descent + unweld + settle
  ├─ [0.1s]   Verification (camera + scene memory + failure detector)
  │
  └─ Total: ~9.5s end-to-end (excl. one-time scene build)
```

## Architecture

```
Qwen3-VL-8B (vLLM on AMD ROCm GPU, ~1.8s)
      │
      ▼
Scene Memory (object registry + position tracking + spatial relations)
      │
      ▼
Task Planner (keyword-based instruction decomposition)
      │
      ▼
Action Scheduler (sequential action dispatch)
      │
      ▼
GraspEnv (unified Gym-style environment)
      ├── Manipulator (Franka Panda, 9 DOF, MJCF)
      ├── Suction Gripper (weld constraint)
      ├── OMPL RRTConnect (collision-free motion planning)
      ├── RigidOptions (Newton solver, box_box_detection)
      └── Stereo Camera (RasterizerCameraOptions, 64×64)
      │
      ▼
Verification Pipeline
      ├── Camera Verify (before/after pixel comparison)
      ├── Scene Memory (position tracking + placement check)
      └── Failure Detector (grasp + placement + disturbance)
```

## Project Structure

```
src/
├── envs/
│   └── grasp_env.py         # GraspEnv — unified Gym-style RL environment
│                              #   - step(): delta-EE actions, 14-dim TensorDict obs
│                              #   - suction_pick/place(): scripted control for demo
│                              #   - keypoint reward: exp(-keypoint_distance)
│                              #   - n_envs parallel support
├── train_grasp.py            # PPO training (rsl-rl-lib, OnPolicyRunner)
├── grasp_bc.py               # Behavior cloning (CNN vision encoder → action)
├── vision/
│   ├── camera.py             # Genesis camera wrapper
│   ├── qwen3vl.py            # Qwen3-VL perception (vLLM OpenAI API + fallback)
│   ├── scene_memory.py       # Object tracking + placement verification
│   └── verifier.py           # Camera before/after pixel verification
├── planner/
│   ├── task_planner.py       # Instruction → action sequence (keyword + optional VLM)
│   ├── action_scheduler.py   # Action sequencing with progress tracking
│   └── recovery.py           # Failure detection + replanning
└── sim/
    └── scene_manager.py      # Genesis scene construction helpers

demo/
├── full_demo.py              # Complete pipeline demo (GraspEnv + VLM + planner)
├── test_e2e.py               # E2E test (all modules wired)
└── output/                   # Demo outputs (images, video, verification.json)

tests/
└── test_recovery_replan.py   # Unit tests for recovery replanning logic
```

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Genesis Init | 0.9s | AMD ROCm GPU, one-time |
| Scene Build | 24s | One-time, compiles GPU kernels |
| Qwen3-VL Inference | 1.8s | Qwen3-VL-8B via vLLM (warm) |
| OMPL Plan Path | ~5s | RRTConnect collision-free |
| Suction Pick | 6.4s | plan_path + weld + PD lift |
| Suction Place | 1.0s | PD descent + unweld + settle |
| Camera Render | 0.3s | 1280×720 RGB |
| Verification | 0.1s | pixel + scene memory + failure detector |
| **End-to-End** | **~9.5s** | Excl. one-time scene build |

## RL Training

GraspEnv follows the official Genesis `grasp_env.py` pattern:

```python
env = GraspEnv(num_envs=64, ctrl_dt=0.01)
obs = env.reset()           # 14-dim: (finger_pos-obj_pos, finger_quat, obj_pos, obj_quat)
obs, reward, done, info = env.step(action)  # 6D delta-EE, keypoint reward
```

Training:
```bash
# Stage 1: RL teacher (PPO, 2048 parallel envs)
python3 src/train_grasp.py -B 2048 --max_iterations 300

# Stage 2: Behavior cloning (stereo RGB → action)
python3 src/grasp_bc.py -B 10 --max_iterations 200
```

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| VLM | Qwen/Qwen3-VL-8B-Instruct | via vLLM 0.25.1 |
| Physics | Genesis | 1.2.2 |
| Robot | Franka Panda (MJCF, panda.xml) | 9 DOF |
| Gripper | Suction (weld constraint) | — |
| RL Framework | rsl-rl-lib | 5.4.2 (PPO) |
| GPU Backend | gs.amdgpu | ROCm 7.2 |
| PyTorch | torch+rocm | 2.11.0 |
| Python | CPython | 3.12 |
| GPU | AMD Radeon Graphics | 48GB VRAM |

## Environment

- AMD Radeon Graphics (48GB VRAM)
- Ubuntu 24.04, ROCm 7.2
- Python 3.12, PyTorch 2.11.0
- Genesis 1.2.2, vLLM 0.25.1
- rsl-rl-lib 5.4.2, tensordict 0.13.0
