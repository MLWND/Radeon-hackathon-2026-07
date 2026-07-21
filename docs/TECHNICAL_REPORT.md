# RoboPilot: Technical Report

## 1. Target Application: Autonomous Desktop Organizer

**Problem:** Modern workspaces accumulate clutter — tools, documents, and objects scattered across desks. Manual sorting is tedious and repetitive.

**Solution:** RoboPilot is a vision-language driven robotic system that understands natural language instructions ("organize my desk", "put the red cup next to the blue box") and autonomously executes multi-step pick-and-place tasks using a Franka Panda robotic arm.

**Target Users:**
- Laboratory technicians organizing reagents and tools
- Warehouse workers sorting packages
- Office workers maintaining clean workspaces
- Researchers demonstrating embodied AI capabilities

**Application Scenarios:**
- **Desk Organization:** "Put the pens in the cup, books on the shelf"
- **Lab Sorting:** "Move the red reagent next to the blue one"
- **Warehouse Picking:** "Pick item A, place in bin B"

---

## 2. System Architecture

```
                    User (Natural Language)
                              │
                    ┌─────────▼─────────┐
                    │   Qwen3-VL-8B     │  ← VLM (vLLM on ROCm GPU)
                    │   Object Detection │
                    │   Spatial Grounding │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Task Planner    │  ← Instruction decomposition
                    │   Action Scheduler│     (keyword + optional VLM)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   GraspEnv        │  ← Unified Gym-style environment
                    │   ├── Manipulator │     (Franka Panda, 9 DOF)
                    │   ├── OMPL Motion │     (RRTConnect collision-free)
                    │   ├── Suction     │     (weld constraint)
                    │   └── Cameras     │     (stereo RGB, 64×64)
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼───┐   ┌──────▼──────┐   ┌────▼────────┐
    │ Scene Memory │   │ Camera      │   │ Failure     │
    │ (positions,  │   │ Verifier    │   │ Detector +  │
    │  relations,  │   │ (pixel diff)│   │ Replanner   │
    │  history)    │   └─────────────┘   └─────────────┘
    └──────────────┘
```

**Key Components:**

| Component | Role | Technology |
|-----------|------|-----------|
| Qwen3-VL-8B | Visual perception, object detection, spatial grounding | vLLM 0.25.1 on ROCm 7.2 |
| TaskPlanner | Instruction → action sequence decomposition | Keyword matching + optional VLM |
| ActionScheduler | Sequential action dispatch with progress tracking | Custom |
| GraspEnv | Unified Gym-style RL environment | Genesis 1.2.2 physics |
| Manipulator | Robot control (IK, PD, motion planning) | Genesis + MJCF Franka Panda |
| SceneMemory | Object tracking, spatial relations, scene graph | Custom |
| CameraVerifier | Before/after pixel verification | NumPy |
| RecoveryManager | Failure detection + automatic replanning | Custom |

---

## 3. Training and Evaluation

### RL Training Pipeline (Stage 1)

**Environment:** GraspEnv with `n_envs` parallel environments on single AMD Radeon GPU.

**Observations (14-dim):**
- `finger_pos - obj_pos` (3D): gripper-to-object offset
- `finger_quat` (4D): gripper orientation
- `obj_pos` (3D): object position
- `obj_quat` (4D): object orientation

**Actions:** 6D delta end-effector `[dx, dy, dz, drx, dry, drz]` × `action_scale=0.05`

**Reward:** Keypoint alignment — `exp(-keypoint_distance)` using 7 keypoints per object (origin + 6 axis-aligned).

**Training:** PPO via rsl-rl-lib 5.4.2, OnPolicyRunner, 2048 parallel environments.

### Behavior Cloning (Stage 2)

**Student Policy:** CNN vision encoder (3 conv layers) → FC → 6D action prediction.

**Input:** Stereo RGB images (6×64×64) + EE pose (7D).

**Data Collection:** RL teacher generates (image, state, action) triples.

**Training:** MSE loss between student and teacher actions.

### Evaluation Metrics

| Metric | Measurement | Result |
|--------|-------------|--------|
| Single-step pick success | Object lifted > 5cm | 100% (demo) |
| Placement accuracy | Euclidean XY error | 2.8cm (demo) |
| Multi-step completion | All steps successful | 100% (4/4 steps) |
| End-to-end latency | VLM + plan + execute + verify | ~9.5s |
| Physics simulation FPS | Genesis on AMD GPU | 404.7 FPS |
| Camera render latency | 640×480 RGB | 6.9ms/frame |

---

## 4. AMD Radeon GPU Utilization

### GPU Workloads

| Stage | GPU Usage | Details |
|-------|-----------|---------|
| VLM Inference | Qwen3-VL-8B via vLLM | ~1.8s per inference, ROCm 7.2 |
| Physics Simulation | Genesis rigid body solver | 404.7 FPS, Newton constraint solver |
| Camera Rendering | Genesis rasterizer | 6.9ms per frame |
| IK Solving | Genesis inverse kinematics | 44.56ms per call |
| RL Training | PPO with 2048 parallel envs | Full GPU utilization |

### ROCm Optimization

- **vLLM on ROCm:** `VLLM_ROCM_USE_AITER=0` (AITER segfaults on RDNA), `--enforce-eager` (torch.compile segfaults on RDNA)
- **Genesis on ROCm:** `gs.amdgpu` backend, automatic kernel compilation and caching
- **PyTorch ROCm:** torch+rocm 2.11.0, HIP backend

### Performance Comparison (GPU vs CPU)

| Metric | AMD GPU (ROCm) | CPU | Speedup |
|--------|---------------|-----|---------|
| Physics FPS | 404.7 | ~20 (est.) | ~20x |
| Camera render | 6.9ms | ~200ms (est.) | ~29x |
| VLM inference | 1.8s | N/A (requires GPU) | — |

---

## 5. Innovations and Key Contributions

### 5.1 VLM-Driven Autonomous Manipulation

Unlike traditional scripted robotics, RoboPilot uses a Vision-Language Model (Qwen3-VL) for:
- **Object detection:** Identifying objects in the scene from natural language
- **Spatial grounding:** Understanding "next to", "left of", "behind"
- **Task decomposition:** Breaking complex instructions into atomic actions

### 5.2 Scene Graph Memory

Maintains a structured scene representation with:
- Object positions and properties (color, type)
- Spatial relations ("left of", "in front of", "near")
- Action history tracking
- Before/after verification

### 5.3 Automatic Failure Recovery

When a grasp or placement fails:
1. **Failure Detection:** Monitors object position changes after each action
2. **Replanning:** Generates new action sequences (not just retries)
3. **Re-execution:** Attempts the replanned actions
4. **Abort:** Stops after too many failures

### 5.4 Unified Gym-Style Environment

GraspEnv provides a standard RL interface:
- `reset()` → 14-dim TensorDict observation
- `step(action)` → (obs, reward, done, info)
- Supports `n_envs` parallel environments
- Compatible with rsl-rl-lib for PPO training

### 5.5 Official Genesis Patterns

All code follows official Genesis examples:
- Ground plane (no kinematic table)
- `RigidOptions(box_box_detection=True, constraint_solver=Newton)`
- `plan_path` + `control_dofs_position` for motion
- Weld constraint for suction grasp
- Stereo camera sensors (`RasterizerCameraOptions`)

---

## 6. Deliverables

| Deliverable | Description |
|-------------|-------------|
| **Source Code** | Complete Python codebase with GraspEnv, VLM pipeline, RL training |
| **Demo Video** | 3-5 minute video showing full workflow |
| **Technical Report** | This document |
| **Reproducibility README** | Setup instructions, dependencies, step-by-step guide |
| **Genesis Example** | VLM-driven pick-and-place example for upstream contribution |
| **Benchmark Results** | GPU vs CPU comparison, performance metrics |

---

## 7. Team Contributions

[To be filled]

---

## References

1. Genesis Physics Engine: https://github.com/Genesis-Embodied-AI/genesis-world
2. Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
3. vLLM: https://github.com/vllm-project/vllm
4. rsl-rl-lib: https://github.com/leggedrobotics/rsl_rl
5. OMPL: https://ompl.kavrakilab.org/
