# RoboPilot Architecture

**Vision-Language Physical AI Robot on AMD GPU**

## Core Principle

- **Qwen3-VL = Brain + Eyes** (understanding + grounding in one model, via vLLM)
- **Genesis = Physics World** (robot, objects, camera, physics on AMD GPU)
- **Suction Gripper = Hand** (weld constraint for reliable pick-and-place)
- **Planner = Spine** (task decomposition + action scheduling)

## Architecture

```
        User Instruction
        "Pick the red cube and place it next to the blue cube"
                    │
                    ▼
          ┌─────────────────┐
          │  Qwen3-VL-8B    │  Brain + Eyes (AMD ROCm GPU)
          │  vLLM Server    │  Image → JSON: {pick, place_relative}
          │  ~1.8s          │  OpenAI-compatible API
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Scene Memory   │  Object Registry
          │  Position Track │  pick_name → entity mapping
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Task Planner   │  Action Decomposition
          │  Action Sched.  │  Instruction → pick → place steps
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
          │  Franka Panda   │  MJCF robot + Ground plane
          │  RigidOptions   │  box_box_detection + Newton solver
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Verify Pipeline │  Multi-method verification
          │ ├ Scene Memory  │  Position tracking + placement check
          │ ├ Camera Verify │  Before/After pixel comparison
          │ └ Fail Detector │  Grasp + placement + disturbance check
          └─────────────────┘
```

## Module Summary

| Module | File | Description |
|--------|------|-------------|
| **SceneManager** | `sim/scene_manager.py` | Genesis scene: ground plane, robot, objects, RigidOptions |
| **CameraWrapper** | `vision/camera.py` | Genesis camera sensor interface |
| **QwenVLWrapper** | `vision/qwen3vl.py` | VLM perception via vLLM (Qwen3-VL-8B) |
| **SceneMemory** | `vision/scene_memory.py` | Object tracking, position updates, placement verification |
| **CameraVerifier** | `vision/verifier.py` | Before/after pixel comparison |
| **TaskPlanner** | `planner/task_planner.py` | Instruction → action decomposition |
| **ActionScheduler** | `planner/action_scheduler.py` | Action sequencing with progress tracking |
| **FailureDetector** | `planner/recovery.py` | Grasp/placement/disturbance failure detection |
| **RecoveryManager** | `planner/recovery.py` | Automatic replanning on failure |
| **ManipulationPipeline** | `control/primitives.py` | Official suction pick/place (plan_path + weld + PD) |

## Pipeline (Verified Working)

```
Step 1: Genesis Init               0.9s  (one-time)
Step 2: Build Scene               24.0s  (one-time, compiles GPU kernels)
Step 3: Qwen3-VL Perception        1.8s  (vLLM server, warm)
Step 4: Suction Pick               6.4s  (OMPL plan_path + weld + PD lift)
Step 5: Suction Place              1.0s  (PD descent + unweld)
Step 6: Verification               0.1s  (scene memory + pixel + fail detector)
───────────────────────────────────────────
Total End-to-End:                 ~9.2s  (excl. one-time setup)
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
- Official Genesis tutorial pattern (suction_cup.py)

### Why Ground Plane Instead of Kinematic Table?

- Kinematic table causes arm clipping during PD control
- Official Genesis tutorials use ground plane + objects at z=half_height
- Ground plane is the standard approach for manipulation benchmarks

### Why OMPL for Approach?

- `plan_path()` (RRTConnect) finds collision-free path above objects
- Works well when arm approaches from above (no table obstacles in upper workspace)
- PD control for descent and placement (smooth, physics-based)

### Why vLLM Instead of Native transformers?

- vLLM provides persistent GPU-resident model (no reload between inferences)
- OpenAI-compatible API (standard interface)
- ~1.8s inference vs ~6s with native transformers
- Same GPU memory footprint as 2B model (369ms vs 3.4s warm)

### Why RigidOptions(Newton)?

- Official Genesis examples use `constraint_solver=gs.constraint_solver.Newton`
- `box_box_detection=True` enables cube-cube collision detection
- More accurate constraint solving for weld constraints

## Interface Contract

### Qwen3-VL Output
```json
{
    "pick": "red_cube",
    "place_relative": "next to the blue cube",
    "reasoning": "The red cube is currently separated from the blue cube"
}
```

### Suction Pick (Official Pattern)
```python
prims.suction_pick("red_cube")
# 1. control_dofs_force([0.5, 0.5]) — open fingers
# 2. plan_path → approach above object
# 3. control_dofs_position — PD descent to grasp height
# 4. rigid_solver.add_weld_constraint(cube_idx, hand_idx) — suction
# 5. control_dofs_position — PD lift
```

### Suction Place (Official Pattern)
```python
prims.suction_place("red_cube", [0.55, 0.0, 0.02])
# 1. control_dofs_position — PD descent to place height
# 2. rigid_solver.delete_weld_constraint(cube_idx, hand_idx) — release
# 3. scene.step(400) — settle
```

## Environment

- **GPU:** AMD Radeon Graphics (48GB VRAM)
- **OS:** Ubuntu 24.04, ROCm 7.2
- **Python:** 3.12
- **PyTorch:** 2.11.0+gitd0c8b1f
- **Genesis:** 1.2.2 (gs.amdgpu backend)
- **vLLM:** 0.25.1+rocm723
- **Transformers:** 5.14.1
- **Model:** Qwen/Qwen3-VL-8B-Instruct (via vLLM)
