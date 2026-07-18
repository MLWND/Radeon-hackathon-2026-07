# RoboPilot Architecture

**Vision-Language Physical AI Robot on AMD GPU**

## Core Principle

- **Qwen3-VL = Brain + Eyes** (understanding + grounding in one model)
- **Genesis = Physics World** (robot, objects, camera, physics on AMD GPU)
- **Suction Gripper = Hand** (weld constraint for reliable pick-and-place)
- **Planner = Spine** (task JSON → action sequence)

## Architecture

```
        User Instruction
        "Pick the red cube and place it next to the blue cube"
                    │
                    ▼
          ┌─────────────────┐
          │   Qwen3-VL-2B   │  Brain + Eyes (AMD ROCm GPU)
          │  Native Class    │  Image → JSON: {pick, place_xyz}
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Task Planner   │  Spine
          │  Object Registry│  pick_name → entity mapping
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  OMPL RRTConnect│  Motion Planning (Genesis built-in)
          │  Collision-free │  plan_path → approach above object
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Suction Grasp  │  Weld Constraint (Genesis rigid_solver)
          │  add_weld_      │  Object attaches to robot hand
          │  constraint()   │  No finger contact needed
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Genesis Scene  │  Physics World (AMD GPU, 200+ FPS)
          │  Franka Panda   │  MJCF robot + Kinematic table
          │  + Camera       │  RGB rendering + depth
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Camera Verify   │  Before/After comparison
          │ Pixel diff +    │  Position tracking
          │ Position check  │  Success confirmation
          └─────────────────┘
```

## Pipeline (Verified Working)

```
Step 1: Load Qwen3-VL           ~14s (one-time)
Step 2: Build Genesis Scene      ~15s (one-time)
Step 3: Camera Capture            317ms
Step 4: Qwen3-VL Perception       ~6s
Step 5: Suction Pick              ~7s (OMPL + weld)
Step 6: Suction Place             ~0.1s (teleport + unweld)
Step 7: Camera Verification       317ms
────────────────────────────────────────────
Total End-to-End:                 ~14s
```

## Key Design Decisions

### Why Suction (Weld Constraint) Instead of Parallel Gripper?

Genesis Franka Panda's parallel gripper has collision geometry issues:
- Finger contact pushes lightweight objects during grasp
- PD controller steady-state error causes arm drift
- Cylinder/curved objects are especially problematic

**Solution:** Use `rigid_solver.add_weld_constraint()` — industry-standard suction approach:
- No finger contact needed
- Object attaches rigidly to hand link
- Reliable for cubes, bottles, any shape
- Official Genesis tutorial pattern

### Why Kinematic Table?

- `gs.materials.Kinematic()` stays fixed during simulation
- Prevents table from falling through ground plane
- Provides stable support surface for objects

### Why OMPL for Approach?

- `plan_path()` (RRTConnect) finds collision-free path above objects
- Works well when arm approaches from above (no table obstacles in upper workspace)
- Teleport used for short-distance descent (safe from nearby positions)

## Module Summary

| Module | File | Status | Description |
|--------|------|--------|-------------|
| Camera | `vision/camera.py` | Done | Genesis camera wrapper |
| Qwen3-VL | `vision/qwen3vl.py` | Done | VLM perception (native Qwen3VLForConditionalGeneration) |
| Primitives | `control/primitives.py` | Done | Suction pick-and-place (weld constraint) |
| Scene | `sim/scene_manager.py` | Done | Genesis scene (Kinematic table + cubes) |
| Orchestrator | `system/orchestrator.py` | Done | Full pipeline |

## Interface Contract

### Qwen3-VL Output
```json
{
    "pick": "red_cube",
    "place_xyz": [0.75, 0.2, 0.07],
    "reasoning": "The red cube is on the table, place near blue cube"
}
```

### Suction Pick
```python
prims.suction_pick("red_cube")
# 1. OMPL plan_path to approach above object
# 2. Teleport to grasp height
# 3. rigid_solver.add_weld_constraint(hand_idx, cube_idx)
# 4. Teleport lift
```

### Suction Place
```python
prims.suction_place("red_cube", [0.75, 0.2, 0.07])
# 1. Teleport above target
# 2. Teleport descend
# 3. rigid_solver.delete_weld_constraint(hand_idx, cube_idx)
# 4. Teleport away
```

## Environment

- **GPU:** AMD Radeon Graphics (48GB VRAM)
- **OS:** Ubuntu 24.04, ROCm 7.2.1
- **Python:** 3.12
- **PyTorch:** 2.9.1+rocm7.2
- **Genesis:** 1.2.2 (gs.amdgpu backend)
- **Transformers:** 5.14.1 (Qwen3VLForConditionalGeneration)
- **Model:** Qwen/Qwen3-VL-2B-Instruct
