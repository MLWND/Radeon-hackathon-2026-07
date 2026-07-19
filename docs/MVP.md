# MVP Development Plan

## Phase 1: Genesis Setup ✅

- [x] Genesis 仿真环境初始化 (GPU backend gs.amdgpu)
- [x] 加载 Franka Panda MJCF 模型 (panda.xml, 9 DOF)
- [x] 基本运动控制 (Genesis IK + PD gains)
- [x] Ground plane + 物体放置 (官方模式，无 kinematic table)
- [x] RigidOptions: box_box_detection + Newton constraint solver

## Phase 2: Robot Grasping ✅

- [x] 尝试 parallel gripper (finger grasp) — 因碰撞问题放弃
- [x] 改用 Suction Gripper (weld constraint) — 稳定工作
- [x] OMPL plan_path 碰撞自由路径规划
- [x] Official suction_cup.py 模式复刻: plan_path → control_dofs_position → weld → lift
- [x] Pick & Place 精度验证: 0.5cm error

## Phase 3: Vision-Language Integration ✅

- [x] Qwen3-VL-8B via vLLM server (OpenAI API, ~1.8s)
- [x] Genesis Camera RGB 渲染 (1280x720)
- [x] VLM 感知 → JSON 任务规划 (pick + place_relative)
- [x] Scene Memory 位置解析 (相对描述 → 精确坐标)
- [x] 端到端 Demo: VLM → pick → place → verify

## Phase 4: Module Integration ✅

- [x] 10 个模块全部串联: SceneManager, Camera, QwenVL, SceneMemory, TaskPlanner, ActionScheduler, ManipPipeline, CameraVerifier, FailDetector, RecoveryManager
- [x] Task Parser: instruction → action decomposition
- [x] Action Scheduler: 按序执行 + 进度跟踪
- [x] Camera Verifier: before/after pixel 对比
- [x] Failure Detector: grasp + placement + disturbance 检查
- [x] Recovery Manager: 失败检测 + 自动重规划

## Phase 5: Polish + Demo ✅

- [x] 完整 E2E 测试脚本 (test_e2e.py, 所有模块参与)
- [x] 摄像头前后对比验证
- [x] 详细日志输出 (时间线 + 模块统计)
- [x] 文档更新 (ARCHITECTURE.md, README.md, MVP.md)
- [x] GitHub Push

## Demo Results (Current)

| Metric | Value |
|--------|-------|
| VLM Model | Qwen/Qwen3-VL-8B-Instruct (via vLLM) |
| VLM Inference | 1.8s |
| Suction Pick | 6.4s (OMPL + weld + PD) |
| Suction Place | 1.0s (PD + unweld) |
| Verification | 0.1s (pixel + scene memory + fail detector) |
| Total End-to-End | 9.2s (excl. one-time setup) |
| Placement Error | 0.5cm |
| Objects Disturbed | 0 |
| Status | **SUCCESS** |

## All Modules Verified

| Module | File | Status | Verified |
|--------|------|--------|----------|
| SceneManager | `sim/scene_manager.py` | ✅ | Ground plane + RigidOptions + Newton |
| CameraWrapper | `vision/camera.py` | ✅ | 1280x720 RGB capture |
| QwenVLWrapper | `vision/qwen3vl.py` | ✅ | vLLM API, 1.8s inference |
| SceneMemory | `vision/scene_memory.py` | ✅ | Position tracking + placement verify |
| CameraVerifier | `vision/verifier.py` | ✅ | Pixel diff + threshold |
| TaskPlanner | `planner/task_planner.py` | ✅ | Rule-based + LLM decomposition |
| ActionScheduler | `planner/action_scheduler.py` | ✅ | Action sequencing + progress |
| ManipulationPipeline | `control/primitives.py` | ✅ | Official suction pick/place |
| FailureDetector | `planner/recovery.py` | ✅ | Grasp + placement + disturbance |
| RecoveryManager | `planner/recovery.py` | ✅ | Auto replanning |
