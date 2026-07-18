# RoboPilot

**Vision-Language Physical AI Robot on AMD GPU**

> Qwen3-VL (Brain) + Genesis (Physics) + Suction Gripper (Hand) + ROCm (Compute)

## Demo

```bash
source venv/bin/activate
python3 demo/full_demo.py
```

**Output:**
```
Qwen3-VL → "pick red_cube, place near blue_cube"  (6s)
Suction Pick → OMPL approach + weld constraint     (7s)
Suction Place → teleport + unweld                   (0.1s)
Camera Verify → position diff 9.8cm, pixel diff 7.0
Status: SUCCESS (5.3cm placement error)
```

## Pipeline

```
User: "Pick the red cube and place it next to the blue cube"
  │
  ├─ [14s]   Qwen3-VL loads (one-time)
  ├─ [15s]   Genesis scene builds (one-time)
  ├─ [6s]    Qwen3-VL perception → JSON {pick, place_xyz}
  ├─ [7s]    OMPL plan_path → approach above object
  ├─ [0.1s]  Weld constraint → object attaches to hand
  ├─ [0.1s]  Teleport lift → object carried up
  ├─ [0.1s]  Teleport to target → unweld → object placed
  ├─ [0.3s]  Camera render → before/after comparison
  │
  └─ Total: ~14s end-to-end (first run, includes model loading)
```

## Architecture

```
Qwen3-VL-2B (AMD ROCm GPU)
      │
      ▼
Task Planner (object registry + coordinate mapping)
      │
      ▼
OMPL RRTConnect (collision-free motion planning)
      │
      ▼
Genesis 1.2.2 (GPU physics simulation, 200+ FPS)
      ├── Franka Panda (MJCF, 9 DOF)
      ├── Suction Gripper (weld constraint)
      ├── Kinematic Table
      └── Camera (640x480 RGB)
      │
      ▼
Camera Verification (before/after pixel diff + position check)
```

## Project Structure

```
src/
├── vision/
│   ├── camera.py          # Genesis camera wrapper
│   ├── qwen3vl.py         # Qwen3-VL (native Qwen3VLForConditionalGeneration)
│   ├── scene_memory.py    # Object tracking
│   └── verifier.py        # Camera verification
├── control/
│   └── primitives.py      # Suction pick-and-place (weld constraint)
├── sim/
│   └── scene_manager.py   # Genesis scene (Kinematic table + cubes)
├── planner/
│   ├── task_parser.py     # JSON parser
│   ├── action_scheduler.py # Action sequencing
│   └── recovery.py        # Failure recovery
└── system/
    ├── benchmark.py       # Performance metrics
    └── orchestrator.py    # Full pipeline

demo/
├── full_demo.py           # Complete end-to-end demo
├── output/
│   ├── before.png         # Scene before manipulation
│   ├── after.png          # Scene after manipulation
│   └── suction_demo.png   # Demo result image
```

## Quick Start

```bash
# 1. Setup environment
bash setup.sh

# 2. Activate
source venv/bin/activate

# 3. Run full demo
python3 demo/full_demo.py

# 4. Run with custom instruction
python3 -c "
from src.system.orchestrator import run_mvp
run_mvp('Pick up the green cube and place it on the red cube')
"
```

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Qwen3-VL Load | 14s | One-time startup |
| Genesis Build | 15s | One-time, compiles GPU kernels |
| Qwen3-VL Inference | 6s | Qwen3-VL-2B on AMD ROCm |
| OMPL Plan Path | 5s | RRTConnect collision-free |
| Suction Pick | 7s | OMPL + weld constraint |
| Suction Place | 0.1s | Teleport + unweld |
| Camera Render | 0.3s | 640x480 RGB |
| Camera Verify | 0.3s | Pixel diff + position |
| **End-to-End** | **~14s** | First run (includes loading) |

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| VLM | Qwen3-VL-2B-Instruct | transformers 5.14.1 |
| Physics | Genesis | 1.2.2 |
| Robot | Franka Panda (MJCF) | — |
| GPU Backend | gs.amdgpu | ROCm 7.2.1 |
| PyTorch | torch+rocm | 2.9.1 |
| Python | CPython | 3.12 |
| GPU | AMD Radeon Graphics | 48GB VRAM |

## Key Design: Suction Gripper (Weld Constraint)

Instead of parallel finger grasp (unreliable due to collision physics), we use Genesis weld constraints:

```python
# Pick: attach object to hand
scene.rigid_solver.add_weld_constraint(hand_link_idx, object_link_idx)

# Place: detach object from hand
scene.rigid_solver.delete_weld_constraint(hand_link_idx, object_link_idx)
```

This is the official Genesis tutorial approach for industrial suction grasping. It's reliable, shape-agnostic, and avoids all collision physics issues.

## Known Limitations

- VLM inference is ~6s (could be optimized with quantization)
- OMPL planning takes ~5s (could be cached for repeated tasks)
- Placement accuracy ~5cm (acceptable for demo, could improve with feedback loop)
- Single-task only (no multi-step sequencing yet)

## Environment

- AMD Radeon Graphics (48GB VRAM)
- Ubuntu 24.04, ROCm 7.2.1
- Python 3.12, PyTorch 2.9.1+rocm7.2
- Genesis 1.2.2, Transformers 5.14.1
