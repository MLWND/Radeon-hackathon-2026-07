# RoboPilot — 视觉语言物理AI机器人

> **Radeon Hackathon 2026-07, Track 3: Physical AI Challenge**

[English](README.md) | 中文

Qwen3-VL-8B (VLM) + Genesis (GPU物理仿真) + 吸盘夹具 (焊接约束) + AMD ROCm

## 演示视频

![演示对比](demo/output/full_comparison.png)

**完整演示视频**: [demo/output/test_e2e.mp4](demo/output/test_e2e.mp4)

## 最终测试结果

| 指标 | 结果 |
|------|------|
| 整体状态 | **SUCCESS** ✅ |
| 放置误差 | **0.3cm** |
| 物体干扰 | **0 个** |
| 视频时长 | **34.4 秒** |
| 帧数 | **1032 帧** |
| VLM推理 | **1.8秒** |
| 抓取时间 | **8.9秒** |
| 放置时间 | **3.4秒** |

## 快速开始

```bash
# 1. 环境设置（一次性）
bash setup.sh

# 2. 运行完整演示
source venv/bin/activate
python3 demo/full_demo.py

# 3. 运行端到端测试（所有10个模块）
python3 demo/test_e2e.py
```

## 功能说明

用户指令：**"拿起红色方块放到蓝色方块旁边"**

```
Qwen3-VL-8B (1.8秒) → 识别 red_cube，规划相对于 blue_cube 的放置位置
Genesis       → OMPL 路径规划，吸盘抓取（焊接约束），PD控制放置
相机          → 像素验证 + 场景记忆 + 故障检测器
```

**完整流水线：约9.2秒端到端（不含一次性设置）**

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| VLM | Qwen/Qwen3-VL-8B-Instruct | vLLM 0.25.1 |
| 物理引擎 | Genesis | 1.2.2 |
| 机器人 | Franka Panda (MJCF) | — |
| 夹具 | 吸盘（焊接约束） | — |
| GPU | AMD Radeon, ROCm 7.2, 48GB VRAM |
| 框架 | PyTorch | 2.11.0 |

## 核心创新

所有代码遵循 Genesis 官方示例模式：

- **使用地面平面**而非运动学桌面（避免机械臂穿模）
- **RigidOptions(Newton, box_box_detection)** 精确碰撞检测
- **焊接约束**吸盘夹具（形状无关，可靠）
- **plan_path → control_dofs_position** 无碰撞接近
- **vLLM** 快速VLM推理（约1.8秒 vs 原生transformers约6秒）

## 系统架构

```
用户: "拿起红色方块放到蓝色方块旁边"
  │
  ├─ Qwen3-VL-8B via vLLM (1.8秒)
  ├─ 场景记忆 → 位置解析
  ├─ 任务规划器 → 动作分解
  ├─ OMPL 路径规划 → 无碰撞接近
  ├─ 吸盘抓取 → 焊接约束 + PD提升 (8.9秒)
  ├─ 吸盘放置 → PD下降 + 解除焊接 (3.4秒)
  └─ 验证 → 像素 + 场景记忆 + 故障检测器 (0.1秒)
```

## 项目结构

```
├── src/                           # 核心源码
│   ├── control/primitives.py      # 机器人控制流水线
│   ├── envs/grasp_env.py          # 抓取环境（核心）
│   ├── planner/
│   │   ├── action_scheduler.py    # 动作调度器
│   │   ├── recovery.py            # 故障恢复（9种故障模式）
│   │   └── task_planner.py        # 任务规划器
│   └── vision/
│       ├── camera.py              # 相机封装
│       ├── qwen3vl.py             # VLM感知
│       ├── scene_memory.py        # 场景记忆
│       └── verifier.py            # 验证器
├── demo/
│   ├── full_demo.py               # 完整闭环演示
│   └── test_e2e.py                # 端到端测试
└── tests/
    └── test_recovery_replan.py    # 故障恢复测试
```

## 故障恢复机制

系统实现了9种故障模式的检测和重试：

| 故障类型 | 检测方法 | 重试策略 |
|----------|----------|----------|
| grasp_failure | 物体没动 | 重新抓取 |
| drop_failure | z坐标下降 | 重新抓取+放置 |
| position_drift | 偏离目标 | 重新抓取+精确放置 |
| ik_failure | IK求解失败 | 调整位置重试 |
| convergence_failure | PD未收敛 | 等待+重试 |
| path_planning_failure | 路径规划失败 | 调整高度重试 |
| weld_constraint_failure | 焊接失败 | 重置+重试 |
| execution_exception | 执行异常 | 恢复+重试 |
| action_failure | 动作失败 | 通用重试 |

## 关键优化

### 执行精度优化
- 使用 Genesis 官方 IK 求解器
- 使用官方抓取高度 (0.130m)
- 放置误差：2.91cm → **0.3cm**

### 物体干扰消除
- 使用 plan_path 碰撞避免功能
- yellow_cylinder 干扰：59cm → **0cm**

### 视频录制修复
- 自动录制每一帧
- 视频时长：1秒 → **34.4秒**

## 文档

- [架构设计](docs/ARCHITECTURE.md)
- [技术报告](docs/TECHNICAL_REPORT.md)
- [最终总结](docs/FINAL_SUMMARY.md)
- [优化报告](docs/OPTIMIZATION_REPORT.md)
- [故障模式覆盖](docs/FAULT_MODE_COVERAGE.md)
- [验证报告](docs/VERIFICATION_REPORT.md)

## 如何申请和使用 AMD Radeon GPU

参见 [README](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/blob/main/Radeon-Cloud-User%20Guide/README.md)

## 提交要求

**请 fork 此仓库并提交 Pull Request**，包含 Luma 页面中规则和条件提到的内容。PR 标题格式："Track x, Team name, your application name"。

> [!NOTE]
> 所有提交材料、项目描述和 Pull Request 都应使用英文提交。

### Track 3 提交要求

1. **技术报告** — 系统架构、AMD GPU 使用、创新点
2. **项目源码** — 完整仓库 + Docker 镜像
3. **可复现性 README** — 环境设置、执行说明
4. **演示视频** — 3-5分钟，AMD GPU 上的完整工作流
5. **补充材料** — PPT / 海报
