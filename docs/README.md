# RoboPilot

**Vision-Language Physical AI Robot on AMD GPU**

> Qwen3-VL-8B (Brain) + Genesis (Physics) + Suction Gripper (Hand) + ROCm (Compute)

## Demo

```bash
# 1. Setup environment (one-time)
bash setup.sh

# 2. Run full demo
source venv/bin/activate
python3 demo/full_demo.py

# 3. Run comprehensive E2E test (all modules)
python3 demo/test_e2e.py
```

**Output:**
```
Qwen3-VL → pick=red_cube, place="next to the blue cube"     (1.8s)
Suction Pick → OMPL plan_path + weld constraint + PD lift    (6.4s)
Suction Place → PD descent + unweld                          (1.0s)
Verify → pixel check + scene memory + fail detector          (0.1s)
Status: SUCCESS (0.5cm placement error, 0 objects disturbed)
```

## Pipeline

```
User: "Pick the red cube and place it next to the blue cube"
  │
  ├─ [0.9s]   Genesis init (AMD ROCm GPU)
  ├─ [24s]    Scene build (one-time, compiles GPU kernels)
  ├─ [1.8s]   Qwen3-VL-8B perception via vLLM → JSON
  ├─ [0.1s]   Task planner → action decomposition
  ├─ [6.4s]   OMPL plan_path → approach → weld → PD lift
  ├─ [1.0s]   PD descent → unweld → settle
  ├─ [0.1s]   Verification (pixel + scene memory + fail detector)
  │
  └─ Total: ~9.2s end-to-end (excl. one-time setup)
```

## Architecture

```
Qwen3-VL-8B (vLLM, AMD ROCm GPU, ~1.8s)
      │
      ▼
Scene Memory (object registry + position tracking)
      │
      ▼
Task Planner (instruction → pick → place steps)
      │
      ▼
OMPL RRTConnect (collision-free motion planning)
      │
      ▼
Genesis 1.2.2 (GPU physics simulation, 200+ FPS)
      ├── Franka Panda (MJCF, 9 DOF)
      ├── Suction Gripper (weld constraint)
      ├── Ground plane (no table, official pattern)
      ├── RigidOptions (Newton solver, box_box_detection)
      └── Camera (1280x720 RGB)
      │
      ▼
Verification Pipeline
      ├── Scene Memory (position tracking + placement check)
      ├── Camera Verify (before/after pixel comparison)
      └── Fail Detector (grasp + placement + disturbance check)
```

## Project Structure

```
src/
├── vision/
│   ├── camera.py          # Genesis camera wrapper
│   ├── qwen3vl.py         # Qwen3-VL perception (vLLM + rule-based fallback)
│   ├── scene_memory.py    # Object tracking + placement verification
│   └── verifier.py        # Camera before/after verification
├── control/
│   └── primitives.py      # ManipulationPipeline (plan_path + weld + PD)
├── sim/
│   └── scene_manager.py   # Genesis scene (ground plane + RigidOptions)
├── planner/
│   ├── task_parser.py     # VLM output → action sequence
│   ├── task_planner.py    # LLM + rule-based task decomposition
│   ├── action_scheduler.py # Action sequencing with progress tracking
│   └── recovery.py        # Failure detection + automatic replanning
└── system/
    └── orchestrator.py    # Full pipeline orchestrator

demo/
├── full_demo.py           # Complete VLM → pick → place → verify demo
├── test_e2e.py            # Comprehensive E2E test (all 10 modules)
├── train_episodes.py      # Multi-episode training loop
└── output/
    ├── visual_before.png  # Scene before manipulation
    ├── visual_after.png   # Scene after manipulation
    └── robopilot_demo.mp4 # Demo video
```

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Genesis Init | 0.9s | AMD ROCm GPU, one-time |
| Scene Build | 24s | One-time, compiles GPU kernels |
| Qwen3-VL Inference | 1.8s | Qwen3-VL-8B via vLLM (warm) |
| OMPL Plan Path | ~5s | RRTConnect collision-free |
| Suction Pick | 6.4s | plan_path + weld + PD lift |
| Suction Place | 1.0s | PD descent + unweld |
| Camera Render | 0.3s | 1280x720 RGB |
| Verification | 0.1s | pixel + scene memory + fail detector |
| **End-to-End** | **~9.2s** | Excl. one-time setup |

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| VLM | Qwen/Qwen3-VL-8B-Instruct | via vLLM 0.25.1 |
| Physics | Genesis | 1.2.2 |
| Robot | Franka Panda (MJCF, panda.xml) | — |
| Gripper | Suction (weld constraint) | — |
| GPU Backend | gs.amdgpu | ROCm 7.2 |
| PyTorch | torch+rocm | 2.11.0+gitd0c8b1f |
| Python | CPython | 3.12 |
| GPU | AMD Radeon Graphics | 48GB VRAM |

## Key Design: Official Genesis Patterns

All code follows official Genesis examples (genesis-world-main/examples/):

```python
# Ground plane only (no kinematic table — causes arm clipping)
scene.add_entity(gs.morphs.Plane())

# RigidOptions for cube-cube collision + Newton solver
gs.options.RigidOptions(box_box_detection=True, constraint_solver=gs.constraint_solver.Newton)

# Official suction pick: plan_path → control_dofs_position → weld → lift
path = robot.plan_path(qpos_goal=qpos_above, num_waypoints=100)
for waypoint in path:
    robot.control_dofs_position(waypoint)
    scene.step()
rigid_solver.add_weld_constraint(cube_idx, hand_idx)

# Official suction place: control_dofs_position → unweld → settle
robot.control_dofs_position(qpos_reach, motors_dof)
rigid_solver.delete_weld_constraint(cube_idx, hand_idx)
```

## Known Limitations

- VLM inference ~1.8s (could be faster with 2B model, but 8B has better reasoning)
- OMPL planning takes ~5s (could be cached for repeated tasks)
- Single-task execution only (official Genesis benchmark pattern)
- Placement accuracy ~0.5cm (acceptable for competition demo)

## Environment

- AMD Radeon Graphics (48GB VRAM)
- Ubuntu 24.04, ROCm 7.2
- Python 3.12, PyTorch 2.11.0+gitd0c8b1f
- Genesis 1.2.2, vLLM 0.25.1, Transformers 5.14.1
