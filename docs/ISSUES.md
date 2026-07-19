# RoboPilot 项目问题清单 (v2)

基于当前代码实际状态的审计。v1 中 6 个 P0 标注为"已修复"，但复查时发现部分 P0 只在某一处修了，未贯穿全项目。本版按实际代码状态重新评估。

**图例：** ✅ 已修复 | 🔧 部分修复 | ❌ 未修复 | 🗑️ 遗留死代码

---

## 一、Robot Control（机器人控制）

### P0-01: IK 没有验证 🔧
**现状：** `primitives.py:92` `solve_ik()` 直接返回 `robot.inverse_kinematics()` 结果，无 FK 验证。
**官方要求：** IK 后用 `get_links_pos()` 验证到位，误差 > 2cm 应 fallback 到 Jacobian IK。
**影响：** IK 静默收敛到错误位置时，抓取完全失败。

### P0-02: PD 控制闭环 + 收敛判断 🔧
**现状：** `primitives.py:131/140` 中 descent/lift 用 hold 模式（设目标一次 + step 50 次）。符合官方"hold"模式，但 **缺收敛检查** — step 50 后无论是否到达都继续。
**官方模式：** hold 是对的，但应在 hold 结束后检查 EE 实际位置与目标差距。
**影响：** 手还没到位就 weld，抓取失败。

### P0-03: Weld constraint 无 force limit ❌
**现状：** weld constraint 是理想吸盘，无 compliance/force limit。
**官方有：** 可加 `control_dofs_force` 限幅或用 `gs.sensors.ContactForce` 监测。
**影响：** 评委可能质疑物理真实性。

### P0-04: 没有 Episode 结构 ❌
**现状：** `ManipulationPipeline` 无 `reset()`/`step(action)`/`is_done()` Gym-style 接口。
**训练用 `train_episodes.py` import 的是旧 `RobotPrimitives`，已失效，无法运行。**
**影响：** 无法 RL、无法多轮训练。

### P0-05: 动作空间是绝对关节角 🔧
**现状：** `solve_ik` → 绝对 qpos → `control_dofs_position`。
**官方推荐：** delta end-effector (6-DOF Cartesian) → `_dls_ik()` → 关节目标。
**影响：** sim-to-real 不友好。

### P1-07: PD Gains 未自调 ✅
直接用官方 tutorial 值，已验证 grasp 精度 0.5-1cm，不需要再调。

### P1-08: rigid_options 已配置 ✅
已在 `scene_manager.py` 和 `full_demo.py` 配置 `box_box_detection=True, constraint_solver=Newton`。

### P1-09: 没有 substeps 🗑️不重要
RigidOptions 已改进，dt=0.01 单步足够稳定。

---

## 二、Perception（感知）

### P1-10: Camera 用 visualization camera 而非 sensor API 🔧
**现状：** `scene.add_camera()`。能工作但不支持批量渲染。
**官方：** `scene.add_sensor(RasterizerCameraOptions)` 可批量。
**影响：** 无法用并行环境。

### P1-11: 没有 Depth Camera ❌
**现状：** 只有 RGB。
**影响：** VLM 空间理解差（无 depth 辅助判断 left/right/behind）。

### P1-12: 相机未 attach 到 robot 🔧
**现状：** 静态第三方相机。官方可挂 robot link。
**影响：** 视角单一。

### P1-13: Stereo camera 已实现但未启用 🔧
**现状：** `primitives.py:53` `setup_stereo_cameras` 方法已写，但 `full_demo.py` 和 `test_e2e.py` 未调用。
**影响：** 单视角 VLM 空间理解不足。

### P1-14: Scene Memory 已实现 ✅
`scene_memory.py` 完整，已用于 position tracking + placement verify。

### P1-15: VLM Prompt 太简单 🔧
**现状：** prompt 给物体列表 + instruction。缺场景描述、空间关系、约束。
**影响：** 复杂指令无法处理。

### P1-16: 没有 Contact Sensor ❌
用 `get_contacts()` 被动读取。无 `gs.sensors.ContactForce` 主动传感器。
**影响：** sim-to-real 验证不专业。

---

## 三、Physics（物理仿真）

### P1-17: Domain Randomization 已实现但未启用 🔧
**现状：** `scene_manager.py:173` `apply_domain_randomization` 已写但 demo 不调用。
**影响：** 策略无法泛化。

### P2-18: 物体属性默认值 ✅不影响

### P2-19: 没有 Parallel Environments ❌
**现状：** 单环境。
**影响：** 训练慢，没利用 GPU 并行。

### P2-20: 没有 performance_mode ❌
未启用 `gs.init(performance_mode=True)`。

---

## 四、Code Architecture（代码架构）— 最高优需要清理

### 🔴 G-01: 遗留死代码（大量失效模块）
**以下文件无任何 import，或被 import 但已失效：**
- `src/control/ik_solver.py` — 无 import (废弃)
- `src/control/gripper.py` — 无 import (废弃)
- `src/control/trajectory.py` — 无 import (废弃)
- `src/planner/task_parser.py` — 无 import (废弃)
- `src/planner/replanner.py` — re-export 但无人用
- `src/sim/robot_wrapper.py` — 无 import (废弃)
- `src/sim/physics_sync.py` — 无 import (废弃)
- `src/system/orchestrator.py` — 无 import (废弃)
- `src/system/orchestrator_v2.py` — 无 import (废弃)
- `src/system/benchmark.py` — 只被废弃 orchestrator 用
- `src/mvp.py` — 无 import (废弃)
- `src/vision/yolo.py` — 无 import (废弃)

**Demo 失效：**
- `demo/headless_demo.py` — import 已不存在的 `RobotPrimitives`
- `demo/train_episodes.py` — import 已不存在的 `RobotPrimitives`
- `demo/visual_demo.py` — import 已不存在的 `RobotPrimitives`
- `demo/test_genesis.py`, `demo/test_robot.py`, `demo/test_yolo.py` — 测试遗留
- `demo/vlm_grounding_demo.py` — 用旧 `create_tabletop_scene`

**影响：** 新人无法判断哪些是活代码，评委看到混乱。

### 🔴 G-02: TaskPlanner/ActionScheduler 与 ManipPipeline 未闭环
**现状：** TaskPlanner 输出 `{"action":"pick", "object":"red_cube"}`，但只用于日志。ManipPipeline 直接用 VLM 的 `pick_name`。ActionScheduler 变摆设。
**影响：** 不是真正的"规划→执行"管线。

### 🟡 G-03: SceneManager 与 Primitives 耦合（已部分解决）
Primitives 直接访问 `scene.sim.rigid_solver`。可接受。

### 🟡 G-04: 没有 Config 系统
所有参数硬编码。可接受 demo 阶段。

### 🟡 G-05: 没有 structured logging
用 print。可接受 demo 阶段。

---

## 五、Demo（演示）

### P2-D1: 没 VLM 输出叠加在视频上 ❌
视频无 pipeline 状态/VLM 决策文字标注。
**影响：** 评委看不懂在干什么。

### P2-D2: 没有三重验证结构化输出 ❌
验证结果 print，无 JSON。
**影响：** 评委无法量化评估。

---

## 修复优先级 (v2)

按对项目质量影响排序：

| 优先级 | 问题 | 修复策略 |
|--------|------|---------|
| **P0-A** | G-01 死代码清理 | 删除废弃模块 + 失效 demo |
| **P0-B** | Demo 失效 (RobotPrimitives) | 删除或重写为 ManipulationPipeline |
| **P0-C** | P0-01 IK 验证 | 加 FK 检查 + Jacobian fallback |
| **P0-D** | P0-02 PD 收敛检查 | hold 结束后检查 EE 位置 |
| **P0-E** | G-02 规划执行闭环 | TaskPlanner/ActionScheduler 真正驱动 ManipPipeline |
| **P0-F** | P0-04 Episode 接口 | 加 reset/step/is_done 到 ManipulationPipeline |
| **P1** | P1-13 启用 stereo camera | full_demo 调用 setup_stereo_cameras |
| **P1** | P1-17 启用 domain randomization | full_demo 调用 apply_domain_randomization |
| **P1** | P2-D1 视频叠加文字 | 用 cv2.putText 在每帧加 VLM 决策 |
| **P1** | P2-D2 三重验证 JSON | 验证时输出结构化 JSON |
