# RoboPilot

**Vision-Language Physical AI Robot on AMD GPU**

> Qwen3-VL (Brain) + Genesis (Physics World) + ROCm (Compute)

## System Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                  RoboPilot                      │
                    │         Vision-Language Physical AI Agent       │
                    └─────────────────────────────────────────────────┘

                                    User Instruction
                                    "Pick up the red cup
                                     and place it in the blue box"
                                            │
                                            ▼
                    ┌───────────────────────────────────┐
                    │         Task Planner              │
                    │   (Qwen3-VL / Smart Parser)       │
                    │                                   │
                    │   Input:  Natural Language        │
                    │   Output: Task Graph (JSON)       │
                    │                                   │
                    │   steps: [                        │
                    │     {action: pick, obj: cup},     │
                    │     {action: place, tgt: box}     │
                    │   ]                               │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │         Scene Memory              │
                    │                                   │
                    │   Tracks: Object positions        │
                    │   Records: Action history         │
                    │   Queries: "Where is the cup?"    │
                    │                                   │
                    │   {cup: [0.19, 0.41, 0.05],       │
                    │    box: [-0.20, 0.15, 0.03]}      │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │       IK Solver + Primitives      │
                    │                                   │
                    │   5 Actions:                      │
                    │   Move / Open / Close / Pick      │
                    │                                   │
                    │   IK: Genesis built-in (GPU)      │
                    │   Control: PD controller          │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │     Genesis Simulation (GPU)      │
                    │                                   │
                    │   Robot:   Franka Panda (MJCF)    │
                    │   Scene:   Tabletop + 4 Objects   │
                    │   Physics: 170+ FPS               │
                    │   Camera:  RGB rendering          │
                    │                                   │
                    │   AMD Radeon PRO W7900D           │
                    │   ROCm 7.2 + PyTorch              │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │       Camera Verification         │
                    │                                   │
                    │   Before: Capture initial state   │
                    │   After:  Capture final state     │
                    │   Verify: Pixel diff + position   │
                    │                                   │
                    │   Result: {success: true,          │
                    │            confidence: 0.97}       │
                    └───────────────────────────────────┘
```

## Pipeline

```
User: "Pick up the red cup and place it in the blue box"
  │
  ├─ [0.2ms]  Task Planner → 2 steps: pick(cup), place(box)
  ├─ [0.5ms]  Scene Memory → cup@[0.19,0.41,0.05], box@[-0.20,0.15,0.03]
  ├─ [5ms]    IK Solver → joint angles (Genesis built-in GPU solver)
  ├─ [3200ms] Genesis → execute pick & place (PD controller)
  ├─ [8ms]    Camera Verify → success=True, confidence=0.97
  │
  └─ Total: ~3.4s
```

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Task Planner | 0.2ms | Rule-based / 3.5s with Qwen3-VL |
| Scene Memory | 0.5ms | Position tracking |
| IK Solver | 5ms | Genesis built-in GPU solver |
| Genesis Step | 2ms | GPU-accelerated |
| Camera Render | 317ms | Headless RGB |
| Camera Verify | 8ms | Pixel diff |
| **Total Pipeline** | **~3.4s** | End-to-end |

## Project Structure

```
src/
├── vision/
│   ├── camera.py          # Genesis camera wrapper
│   ├── qwen3vl.py         # Qwen3-VL VLM
│   ├── scene_memory.py    # Object tracking
│   └── verifier.py        # Camera verification
├── planner/
│   ├── task_planner.py    # Task graph generation
│   ├── task_parser.py     # JSON parser
│   ├── action_scheduler.py # Action sequencing
│   └── recovery.py        # Failure recovery
├── control/
│   ├── primitives.py      # 5 basic actions (PD control)
│   ├── ik_solver.py       # Standalone IK (reference)
│   ├── trajectory.py      # Path generation (reference)
│   └── gripper.py         # Gripper control (reference)
├── sim/
│   ├── scene_manager.py   # Genesis scene (MJCF Franka, tabletop)
│   ├── robot_wrapper.py   # Robot interface
│   └── physics_sync.py    # Physics stepping
└── system/
    ├── benchmark.py       # Performance metrics
    ├── orchestrator.py    # MVP V1
    └── orchestrator_v2.py # Full pipeline (V2)

benchmark/
├── gpu_test.py            # GPU matrix benchmark
├── parallel_sim.py        # Parallel simulation
└── generate_charts.py     # Paper-quality charts
```

## Quick Start

```bash
# 1. One-click environment recovery (after cloud instance restart)
bash setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run full MVP V2
python3 -c "from src.system.orchestrator_v2 import run_mvp_v2; run_mvp_v2('Pick up the red cup and place it in the blue box')"

# 4. Run multi-step task
python3 -c "from src.system.orchestrator_v2 import run_mvp_v2; run_mvp_v2('Pick up the apple and place it in the blue box, then pick up the red cup')"

# 5. Run parallel benchmark
python3 benchmark/parallel_sim.py

# 6. Run benchmark charts
python3 benchmark/generate_charts.py
```

## Known Issues

### 🔴 Critical: Grasp Not Working (PD Control)

**Status:** Under investigation

**Symptom:** Robot arm reaches the cup position via IK + PD control, gripper closes, but the cup is pushed sideways rather than lifted. Verification correctly reports FAILED.

**Root Cause Analysis:**

1. **IK frame mismatch**: Genesis `inverse_kinematics(hand_link, target)` reports ~1e-7 error (converged), but `set_qpos(IK_result)` places the hand 3-5cm away from the target. This suggests the IK's internal FK and `get_links_pos()` use different coordinate frames or link definitions.

2. **Finger joint geometry**: MJCF Franka fingers are prismatic joints along the Y-axis (not Z). Fingers extend alongside the hand, not below it. The grasp relies on horizontal finger closure, requiring precise X/Y alignment.

3. **Tendon approximation**: Original `panda.xml` uses a tendon-driven gripper. Genesis approximates this as joint actuators, but `control_dofs_position` doesn't reliably drive the gripper. Switched to `panda_no_tendon.xml` which works but may have different kinematics.

4. **PD controller convergence**: `control_dofs_position` from home (all-zeros) to IK target has steady-state error. The robot needs more simulation steps to converge.

**Fixes Required:**

- Investigate Genesis IK link frame vs `get_links_pos` frame mismatch
- Verify `set_qpos(IK_result)` produces correct hand position
- Tune PD gains or increase convergence steps
- Possibly use `control_dofs_force` for gripper closing
- Test with known-good URDF model instead of MJCF

**Current Workaround:** The system correctly detects and reports grasp failure (moved=True, near=False). Recovery retries but cannot succeed until the IK/convergence issue is resolved.

### 🟡 Minor: Ready Pose Knocks Objects

**Status:** Known, low priority

**Symptom:** `_go_to_ready_pose()` drives the arm from home to a bent position, which can collide with and push tabletop objects during scene settling.

**Fix:** Disable `_go_to_ready_pose()` in `scene_manager.py` or increase settle distance.

## Environment

- AMD Radeon PRO W7900D (48GB VRAM)
- ROCm 7.2
- PyTorch 2.9.1+rocm7.2
- Genesis 1.1.2
- Qwen3-VL-2B (optional)

## License

MIT
