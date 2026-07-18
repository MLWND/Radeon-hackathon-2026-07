# RoboPilot — Vision-Language Physical AI Robot

> **Radeon Hackathon 2026-07, Track 3: Physical AI Challenge**

Qwen3-VL (VLM) + Genesis (GPU Physics) + Suction Gripper (Weld Constraint) + AMD ROCm

## Quick Start

```bash
# 1. Setup (one-time)
bash setup.sh

# 2. Run full demo
source venv/bin/activate
python3 demo/full_demo.py
```

## What It Does

User says: **"Pick the red cube and place it next to the blue cube"**

```
Qwen3-VL  →  identifies red_cube, plans placement
Genesis   →  OMPL path planning, suction pick, place
Camera    →  verifies before/after, confirms success
```

**Full pipeline: ~14s end-to-end on AMD ROCm GPU**

## Demo Results

| Metric | Value |
|--------|-------|
| VLM Model | Qwen3-VL-2B (native Qwen3VLForConditionalGeneration) |
| VLM Inference | 6s |
| Suction Pick | 7s |
| Suction Place | 0.1s |
| Placement Error | 5.3cm |
| Status | **SUCCESS** |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| VLM | Qwen/Qwen3-VL-2B-Instruct |
| Physics | Genesis 1.2.2 (gs.amdgpu) |
| Robot | Franka Panda (MJCF) |
| Gripper | Suction (weld constraint) |
| GPU | AMD Radeon, ROCm 7.2, 48GB VRAM |
| Framework | PyTorch 2.9.1+rocm7.2 |

## Key Innovation: Suction Gripper via Weld Constraint

Instead of unreliable parallel finger grasp, we use Genesis weld constraints — the official industrial approach:

```python
# Pick: attach object to hand
scene.rigid_solver.add_weld_constraint(hand_link, object_link)

# Place: detach object
scene.rigid_solver.delete_weld_constraint(hand_link, object_link)
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
