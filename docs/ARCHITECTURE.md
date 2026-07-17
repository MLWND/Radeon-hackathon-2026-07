# RoboPilot Architecture

**Vision-Language Robot — Single Perception Backbone**

## Core Principle

- **Qwen3-VL = Brain + Eyes** (understanding + grounding in one model)
- **Genesis = Physics World** (robot, objects, camera, physics)
- **Planner = Spine** (task JSON → action sequence)
- **IK = Muscles** (Cartesian goals → joint angles)

**YOLO is NOT in the core pipeline.** It exists only as an optional fast-mode backend.

## Why No YOLO?

Qwen3-VL has built-in **Object Grounding** — it can directly output bounding boxes and points for detected objects. This eliminates the need for a separate detector.

| | YOLO + LLM | Qwen3-VL Only |
|---|---|---|
| Models | 2 (YOLO + LLM) | 1 (Qwen3-VL) |
| Pipeline | RGB → YOLO → bbox → LLM → plan | RGB → Qwen3-VL → bbox + plan |
| Innovation | Common approach | Novel, showcases VLM |
| Maintenance | Two model dependencies | Single model |

## Architecture

```
            User
  "把左边的红色杯子放到蓝色盒子"
              │
              ▼
        ┌─────────────┐
        │  Qwen3-VL   │  Brain + Eyes
        │  (AMD GPU)  │  Detect + Ground + Reason
        └──────┬──────┘
               │ JSON + Bounding Box
               ▼
        ┌─────────────┐
        │   Planner   │  Spine
        │  (CPU/GPU)  │  Task → Action Sequence
        └──────┬──────┘
               │ Action List
               ▼
        ┌─────────────┐
        │  IK Solver  │  Muscles
        │  (AMD GPU)  │  xyz → Joint Angles
        └──────┬──────┘
               │ Joint Targets
               ▼
        ┌─────────────┐
        │  Controller │  Nerves
        │  (AMD GPU)  │  Joint → Motor Commands
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Genesis   │  Physics World
        │  (AMD GPU)  │  Simulation Step
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
     Robot        Camera
    (Franka)     (RGB Image)
                       │
                       │ (optional fast mode)
                       ▼
                  ┌─────────┐
                  │  YOLO   │  Fast Eyes (optional)
                  │ (AMD GPU)│  High-frequency detection
                  └─────────┘
```

## Module List (12 modules — simplified from 15)

### Layer 1: Perception

| # | Module | Description | Input | Output | Status |
|---|--------|-------------|-------|--------|--------|
| 1 | `vision.camera` | Genesis camera wrapper | scene | RGB image | Done |
| 2 | `vision.qwen3vl` | VLM: detect + ground + reason | image + text | JSON + bbox | **Core** |
| 3 | `vision.scene_memory` | Track objects across frames | detections | object map | Done |
| ~~4~~ | ~~`vision.yolo`~~ | ~~Object detection~~ | ~~RGB~~ | ~~detections~~ | **Optional fast-mode** |

### Layer 2: Planning

| # | Module | Description | Input | Output | Status |
|---|--------|-------------|-------|--------|--------|
| 4 | `planner.task_parser` | Parse VLM output to actions | JSON | action list | Done |
| 5 | `planner.action_scheduler` | Sequence actions | action list | execution plan | Done |
| 6 | `planner.replanner` | Re-plan on failure | error + state | new plan | Done |

### Layer 3: Control

| # | Module | Description | Input | Output | Status |
|---|--------|-------------|-------|--------|--------|
| 7 | `control.ik_solver` | Inverse kinematics | xyz | joint angles | Done |
| 8 | `control.trajectory` | Trajectory generation | waypoints | trajectory | Done |
| 9 | `control.gripper` | Gripper control | command | state | Done |

### Layer 4: Simulation

| # | Module | Description | Input | Output | Status |
|---|--------|-------------|-------|--------|--------|
| 10 | `sim.scene_manager` | Genesis scene setup | config | scene | Done |
| 11 | `sim.robot_wrapper` | Robot interface | targets | state | Done |
| 12 | `sim.physics_sync` | Physics stepping | commands | state | Done |

### Layer 5: System

| # | Module | Description | Input | Output | Status |
|---|--------|-------------|-------|--------|--------|
| 13 | `system.benchmark` | Performance metrics | pipeline | report | Done |
| 14 | `system.orchestrator` | Main pipeline loop | input | result | Done |

## Interface Contracts

### Qwen3-VL → Planner
```python
# Qwen3-VL output (with grounding)
{
    "task": "pick_place",
    "reasoning": "User wants to move the red cup to the blue box",
    "object": {
        "type": "cup",
        "color": "red",
        "bbox": [120, 180, 250, 320],
        "center_pixel": [185, 250],
        "confidence": 0.92
    },
    "target": {
        "type": "box",
        "color": "blue",
        "bbox": [400, 200, 520, 350],
        "center_pixel": [460, 275],
        "confidence": 0.88
    }
}
```

### Planner → IK
```python
# Action sequence
[
    {"action": "move_to", "xyz": [0.4, 0.1, 0.15]},
    {"action": "move_down", "xyz": [0.4, 0.1, 0.05]},
    {"action": "close_gripper"},
    {"action": "lift", "height": 0.15},
    {"action": "move_to", "xyz": [-0.3, 0.2, 0.15]},
    {"action": "move_down", "xyz": [-0.3, 0.2, 0.05]},
    {"action": "open_gripper"},
    {"action": "lift", "height": 0.15}
]
```

## Dual Mode (Optional Enhancement)

| Mode | Pipeline | Use Case |
|------|----------|----------|
| **Reasoning Mode** | Qwen3-VL → Plan → Execute | Complex tasks, natural language, spatial reasoning |
| **Fast Mode** | YOLO → Plan → Execute | High-frequency, fixed categories, real-time |

The two modes can be switched at runtime. Fast Mode is an optimization, not a replacement.

## Directory Structure

```
AMD_PhysicalAI/
├── src/
│   ├── vision/
│   │   ├── camera.py          # Module 1
│   │   ├── qwen3vl.py         # Module 2 (CORE)
│   │   ├── scene_memory.py    # Module 3
│   │   └── yolo.py            # Optional fast-mode
│   ├── planner/
│   │   ├── task_parser.py     # Module 4
│   │   ├── action_scheduler.py # Module 5
│   │   └── replanner.py       # Module 6
│   ├── control/
│   │   ├── ik_solver.py       # Module 7
│   │   ├── trajectory.py      # Module 8
│   │   └── gripper.py         # Module 9
│   ├── sim/
│   │   ├── scene_manager.py   # Module 10
│   │   ├── robot_wrapper.py   # Module 11
│   │   └── physics_sync.py    # Module 12
│   └── system/
│       ├── benchmark.py       # Module 13
│       └── orchestrator.py    # Module 14
├── tests/
├── benchmark/
├── demo/
├── docs/
└── README.md
```

## Performance Targets

| Component | Target Latency |
|-----------|---------------|
| Qwen3-VL (detect + ground + reason) | < 200ms |
| Task Planning | < 50ms |
| IK Solve | < 5ms |
| Trajectory Gen | < 10ms |
| Genesis Step | < 2ms |
| **End-to-End (Reasoning Mode)** | **< 300ms** |
| YOLO (Fast Mode, optional) | < 10ms |
| **End-to-End (Fast Mode)** | **< 50ms** |
