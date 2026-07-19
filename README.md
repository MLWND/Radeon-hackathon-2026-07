# RoboPilot — Vision-Language Physical AI Robot

> **Radeon Hackathon 2026-07, Track 3: Physical AI Challenge**

Qwen3-VL-8B (VLM) + Genesis (GPU Physics) + Suction Gripper (Weld Constraint) + AMD ROCm

## Quick Start

```bash
# 1. Setup (one-time)
bash setup.sh

# 2. Run full demo
source venv/bin/activate
python3 demo/full_demo.py

# 3. Run comprehensive E2E test (all 10 modules)
python3 demo/test_e2e.py
```

## What It Does

User says: **"Pick the red cube and place it next to the blue cube"**

```
Qwen3-VL-8B (1.8s) → identifies red_cube, plans placement relative to blue_cube
Genesis       → OMPL plan_path, suction pick (weld), PD place
Camera        → pixel verify + scene memory + fail detector
```

**Full pipeline: ~9.2s end-to-end on AMD ROCm GPU (excl. one-time setup)**

## Demo Results

| Metric | Value |
|--------|-------|
| VLM Model | Qwen3-VL-8B-Instruct (via vLLM 0.25.1) |
| VLM Inference | 1.8s |
| Suction Pick | 6.4s (OMPL + weld + PD lift) |
| Suction Place | 1.0s (PD descent + unweld) |
| Placement Error | 0.5cm |
| Objects Disturbed | 0 |
| Status | **SUCCESS** |

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| VLM | Qwen/Qwen3-VL-8B-Instruct | vLLM 0.25.1 |
| Physics | Genesis | 1.2.2 |
| Robot | Franka Panda (MJCF) | — |
| Gripper | Suction (weld constraint) | — |
| GPU | AMD Radeon, ROCm 7.2, 48GB VRAM |
| Framework | PyTorch | 2.11.0+gitd0c8b1f |

## Key Innovation: Official Genesis Patterns

All code follows official Genesis examples. Key patterns:

- **Ground plane** instead of kinematic table (avoids arm clipping)
- **RigidOptions(Newton, box_box_detection)** for accurate collision
- **Weld constraint** suction gripper (shape-agnostic, reliable)
- **plan_path → control_dofs_position** for collision-free approach
- **vLLM** for fast VLM inference (~1.8s vs ~6s with native transformers)

## Architecture

```
User: "Pick the red cube and place it next to the blue cube"
  │
  ├─ Qwen3-VL-8B via vLLM (1.8s)
  ├─ Scene Memory → Position Resolution
  ├─ Task Planner → Action Decomposition
  ├─ OMPL Plan Path → Collision-free Approach
  ├─ Suction Pick → Weld Constraint + PD Lift (6.4s)
  ├─ Suction Place → PD Descent + Unweld (1.0s)
  └─ Verification → Pixel + Scene Memory + Fail Detector (0.1s)
```

## Project Docs

- [Architecture](docs/ARCHITECTURE.md) — system design and module list
- [MVP Plan](docs/MVP.md) — development timeline and results
- [Technical README](docs/README.md) — detailed pipeline and performance

---

## How to Apply and Use AMD Radeon GPU

See [README](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/blob/main/Radeon-Cloud-User%20Guide/README.md)

## When You Submit

**Please fork this repo and open a pull request** including the stuff mentioned in Rules & Conditions of the Luma page. The title of the pull request should be like "Track x, Team name, your application name".

> [!NOTE]
> All submission materials, project descriptions, and Pull Requests should be submitted in English.

## Submission Requirements (Track 3)

1. **Technical Report** — system architecture, AMD GPU utilization, innovations
2. **Project Source Code** — complete repository + Docker image
3. **Reproducibility README** — environment setup, execution instructions
4. **Demo Video** — 3-5 minutes, complete workflow on AMD GPU
5. **Supplementary materials** — PPT / Poster
