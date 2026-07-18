# RoboPilot 项目问题清单

基于 Genesis 官方文档的全面审计。按模块分类，标注优先级和影响。

---

## 一、Robot Control（机器人控制）

### P0-01: IK 没有验证
**现状：** `solve_ik()` 返回 joint angles 后直接使用，不检查 FK 是否到达目标。
**文档要求：** 官方示例用 FK 验证 IK 结果。
```python
# 缺失
qpos = robot.inverse_kinematics(link=ee, pos=target)
robot.set_qpos(qpos)
hand_pos = robot.get_links_pos()[hand_link_idx]
assert np.linalg.norm(hand_pos - target) < threshold
```
**影响：** 可能抓偏，手没有到达预期位置。

### P0-02: PD 控制没有闭环
**现状：** `control_dofs_position(target)` 只调用一次，然后 step 300 次。
**文档要求：** 官方 locomotion/manipulation 示例每个 step 都设 target。
```python
# 正确做法
for i in range(steps):
    robot.control_dofs_position(target_qpos)
    scene.step()
```
**影响：** PD 稳态误差累积，手偏离目标。

### P0-03: Weld Constraint 不是标准抓取方式
**现状：** 用 `rigid_solver.add_weld_constraint()` 做吸盘抓取。
**文档要求：** 官方 manipulation 示例用 parallel gripper + scripted close-and-lift。
**影响：** 评委可能质疑物理真实性。吸盘是理想化模型，没有 compliance/force limit。

### P0-04: 没有 Episode 结构
**现状：** 没有 reset/step/done 流程。
**文档要求：** 所有 RL 示例都有完整的 Gym-style 接口。
```python
def reset(self): ...
def step(self, actions): ...  # returns (obs, reward, done, info)
```
**影响：** 无法做 RL 训练，也无法做多轮演示。

### P0-05: 动作空间不规范
**现状：** 直接设置关节角度（absolute joint position）。
**文档要求：** 官方 manipulation 用 delta end-effector command (6-DOF)。
```python
# 官方做法
action = [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz]
q_pos = self._dls_ik(action)  # Cartesian delta → joint targets
self.robot.control_dofs_position(position=q_pos)
```
**影响：** 对 sim-to-real 不友好，real robot 不会用绝对关节角。

### P0-06: 没有 Action Scaling
**现状：** 直接设绝对位置，无幅度限制。
**文档要求：** 官方有 `action_scales = 0.05` per step。
**影响：** 动作过大可能导致不稳定。

### P1-07: PD Gains 可能不适合当前场景
**现状：** 用 tutorial 默认值 `[4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]`。
**文档要求：** 官方说 "always recommended to tune it manually or refer to a well-tuned value online"。
**影响：** PD 收敛慢或振荡。

### P1-08: rigid_options 没配置
**现状：** 没有设置碰撞检测参数。
**文档要求：**
```python
rigid_options=gs.options.RigidOptions(
    enable_collision=True,
    enable_joint_limit=True,
    constraint_solver=gs.constraint_solver.Newton,
    max_collision_pairs=20,
)
```
**影响：** 物理仿真不准确。

### P1-09: 没有 substeps
**现状：** 只用默认 dt=0.01。
**文档要求：** locomotion 示例用 `substeps=2` 提高稳定性。
**影响：** 快速运动时物理不稳定。

---

## 二、Perception（感知）

### P1-10: Camera 用法错误
**现状：** `scene.add_camera()` (visualization camera)。
**文档要求：** 正式渲染应该用 `scene.add_sensor(gs.sensors.RasterizerCameraOptions(...))`。
**影响：** 不支持批量渲染，不支持 sensor pipeline（noise/delay/history）。

### P1-11: 没有 Depth Camera
**现状：** 只有 RGB。
**文档有：** `gs.sensors.DepthCamera` 可以直接给 VLM 提供深度信息。
```python
depth_cam = scene.add_sensor(gs.sensors.DepthCamera(
    pattern=gs.sensors.DepthCameraPattern(res=(96, 72), fov_horizontal=90.0),
    entity_idx=robot.idx, link_idx_local=0))
```
**影响：** VLM 无法获取深度信息，空间理解差。

### P1-12: 相机没有 attach 到 robot
**现状：** 静态相机在固定位置。
**文档要求：** 可以把相机挂在 robot link 上，跟随运动。
```python
camera = scene.add_sensor(..., entity_idx=robot.idx, link_idx_local=0)
```
**影响：** 视角单一，无法观察抓取过程。

### P1-13: 只有一个相机视角
**官方 manipulation 用 stereo camera：**
```python
rgb = env.get_stereo_rgb_images(normalize=True)  # (n_envs, 6, H, W)
```
**影响：** 单视角 VLM 空间理解不足。

### P1-14: 没有 Scene Memory
**文档有 `scene_memory` 模块追踪物体位置。** 我们每次重新识别。
**影响：** 无法跟踪物体状态变化。

### P1-15: VLM Prompt 太简单
**现状：** `"Pick the red cube and place it in the blue goal area"`
**应该提供：** 场景描述、物体属性、空间关系、约束条件。
**影响：** VLM 无法处理复杂指令。

### P1-16: 没有 Contact Sensor（Sensor API）
**现状：** `target_obj.get_contacts()` 一次性读取。
**文档要求：** 用 `gs.sensors.ContactForce` 传感器，支持 noise/delay/history。
```python
contact_sensor = scene.add_sensor(gs.sensors.ContactForce(
    entity_idx=robot.idx, link_idx_local=hand_link_idx_local,
    history_length=4))
```
**影响：** 验证不专业，无法做 sim-to-real。

---

## 三、Physics（物理仿真）

### P1-17: 没有 Domain Randomization
**文档强调这是 sim-to-real 关键。**
```python
robot.set_friction_ratio(0.5 + torch.rand(n_envs, n_links))
robot.set_mass_shift(-0.5 + torch.rand(n_envs, n_links))
robot.set_dofs_kp(4000 + 1000 * torch.rand(n_envs, n_dofs), ...)
```
**影响：** 策略无法泛化到真实世界。

### P2-18: 物体属性没调优
**现状：** 摩擦力、质量、阻尼都是默认值。
**影响：** 抓取行为可能不稳定。

### P2-19: 没有 Parallel Environments
**官方推荐：**
```python
scene.build(n_envs=4096)  # 并行环境
```
**影响：** 没有利用 GPU 并行能力，演示速度慢。

### P2-20: 没有 performance_mode
**官方推荐训练时用：**
```python
gs.init(backend=gs.gpu, performance_mode=True)
```
**影响：** 编译慢，step 性能差。

---

## 四、Sensing（传感器）

### P1-21: 没有 IMU 传感器
**文档有 `gs.sensors.IMU`。** 可以给 VLM 提供运动状态。
**影响：** VLM 缺少运动上下文。

### P1-22: 没有 Temperature Grid
**文档有 `gs.sensors.TemperatureGrid`。** 可以展示高级感知能力。
**影响：** 演示不够丰富。

### P2-23: 没有 Raycaster / Lidar
**文档有 `gs.sensors.Lidar`。** 可以做环境感知。
**影响：** 演示场景单一。

### P2-24: 没有 Proximity Sensor
**文档有 `gs.sensors.SurfaceDistanceProbe`。** 可以检测物体距离。
**影响：** 缺少距离感知能力。

---

## 五、Rendering（渲染）

### P1-25: 没有使用 Nyx Renderer
**文档推荐 Nyx 做 photorealistic rendering。** 我们只用 rasterizer。
**影响：** 渲染质量不够专业。

### P2-26: 光照配置太简单
**现状：** 只有 1 个 DirectionalLight + 1 个 PointLight + ambient。
**文档有：** 更复杂的光照配置，包括 HDRI environment maps。
**影响：** 阴影偏硬，视觉效果一般。

### P2-27: 没有使用 GS surffaces 的 PBR 材质
**现状：** 用 `gs.surfaces.Smooth(color=..., roughness=0.3)`。
**文档有：** `Metal`、`Glass`、`Emission` 等高级材质。
**影响：** 物体质感不够真实。

---

## 六、Recording & Verification（录制与验证）

### P2-28: 视频录制方式不规范
**现状：** `camera.start_recording()` + 手动 render。
**文档推荐：** `scene.start_recording()` + `gs.recorders.VideoFile`。
**影响：** 录制质量差，帧率不稳定。

### P2-29: 没有 Performance Benchmark
**文档有：** `vllm bench serve`、Genesis 内置 benchmark。
**影响：** 无法展示 GPU 加速性能。

### P2-30: 没有三重验证的结构化输出
**现状：** 验证结果只是 print。
**应该：** 输出结构化 JSON，包含所有验证指标。
**影响：** 评委无法量化评估。

### P2-31: 没有 before/after 对比图
**现状：** 手动保存 before/after。
**应该：** 用 `cv2` 合成对比图，标注差异。
**影响：** 演示效果差。

---

## 七、Code Architecture（代码架构）

### P2-32: RobotPrimitives 职责不清
**现状：** 既管 IK 又管 grasp 又管 move 又管 render callback。
**应该：** 拆分为 `IKSolver`、`GraspController`、`MotionPlanner`。

### P2-33: SceneManager 和 Primitives 耦合太紧
**现状：** Primitives 直接访问 `scene.rigid_solver`。
**应该：** 通过 SceneManager 抽象层访问。

### P2-34: 没有 Config 系统
**现状：** 所有参数硬编码（PD gains、camera pose、物体位置等）。
**应该：** 用 dataclass 或 YAML 配置文件。
**影响：** 改参数需要改代码。

### P2-35: 没有 Logging
**现状：** 只有 `print()`。
**应该：** 用 Python `logging` 模块，structured logging。
**影响：** 调试困难。

### P2-36: 没有 Error Handling
**现状：** IK 失败、grasp 失败等没有 try-catch 或恢复逻辑。
**影响：** 程序容易崩溃。

### P2-37: 没有 Type Hints
**现状：** 很多函数没有类型标注。
**影响：** IDE 支持差，容易出错。

---

## 八、Demo/展示（演示）

### P2-38: VLM 推理时间太长
**现状：** Qwen3-VL-8B ~3.3s。
**应该：** 用更小模型或量化加速到 <1s。
**影响：** 演示节奏慢。

### P2-39: 没有多步任务演示
**现状：** 只有 "Pick red, place at goal"。
**应该：** 演示 "先拿红的放到蓝的旁边，再拿绿的放到桌子中间"。
**影响：** VLM 的价值没有充分体现。

### P2-40: 没有 Recovery 机制
**现状：** 抓取失败后没有重试。
**应该：** 检测失败 → 调整位置 → 重试。
**影响：** 演示可靠性差。

### P2-41: 目标区域太小
**现状：** Goal area 12cm × 12cm。
**应该：** 更大的目标区域，容错更高。
**影响：** 放置精度要求过高。

### P2-42: 没有文字标注在视频上
**现状：** 视频没有叠加 VLM 输出、任务状态等信息。
**应该：** 每帧叠加 pipeline 状态、VLM 决策、验证结果。
**影响：** 评委看不懂在干什么。

---

## 总结

| 类别 | P0 | P1 | P2 | 总计 |
|------|-----|-----|-----|------|
| Robot Control | 6 | 3 | 0 | 9 |
| Perception | 0 | 7 | 0 | 7 |
| Physics | 0 | 2 | 2 | 4 |
| Sensing | 0 | 2 | 2 | 4 |
| Rendering | 0 | 1 | 2 | 3 |
| Recording | 0 | 0 | 4 | 4 |
| Code Architecture | 0 | 0 | 6 | 6 |
| Demo | 0 | 0 | 5 | 5 |
| **总计** | **6** | **13** | **19** | **38** |

**最紧急的 6 个 P0 问题：**
1. IK 没验证 → 抓偏
2. PD 没闭环 → 控制不精确
3. Weld constraint 不标准 → 评委质疑
4. 没有 Episode 结构 → 无法 RL
5. 动作空间不规范 → sim-to-real 失败
6. 没有 Action Scaling → 不稳定
