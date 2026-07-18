# MVP Development Plan

## Week 1: Genesis Setup ✅

- [x] Genesis 仿真环境初始化 (GPU backend gs.amdgpu)
- [x] 加载 Franka Panda MJCF 模型
- [x] 基本运动控制 (IK + PD)
- [x] Kinematic 桌面 + 物体放置

## Week 2: Robot Grasping ✅

- [x] 尝试 parallel gripper (finger grasp) — 因碰撞问题放弃
- [x] 改用 Suction Gripper (weld constraint) — 稳定工作
- [x] OMPL plan_path 碰撞自由路径规划
- [x] Pick & Place Demo 成功

## Week 3: Vision-Language Integration ✅

- [x] Qwen3-VL 模型加载 (native Qwen3VLForConditionalGeneration)
- [x] Genesis Camera RGB 渲染
- [x] VLM 感知 → JSON 任务规划
- [x] 端到端 Demo: "Pick red_cube, place near blue_cube"

## Week 4: Polish + Demo ✅

- [x] 摄像头前后对比验证
- [x] 性能指标统计
- [x] 文档更新
- [x] GitHub Push

## Demo Results

| Metric | Value |
|--------|-------|
| VLM Model | Qwen/Qwen3-VL-2B-Instruct |
| VLM Inference | ~6s |
| Suction Pick | ~7s |
| Suction Place | ~0.1s |
| Total End-to-End | ~14s |
| Placement Error | 5.3cm |
| Status | SUCCESS |

## Files Modified

| File | Description |
|------|-------------|
| `src/vision/qwen3vl.py` | Qwen3-VL wrapper (native class) |
| `src/control/primitives.py` | Suction pick-and-place (weld constraint) |
| `src/sim/scene_manager.py` | Scene: Kinematic table + cubes |
| `src/system/orchestrator.py` | Full pipeline orchestrator |
