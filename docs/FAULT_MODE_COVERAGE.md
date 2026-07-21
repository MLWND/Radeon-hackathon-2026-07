# 机械臂抓取放置逻辑故障模式覆盖报告

## 概述

本报告详细分析了机械臂抓取放置流程中所有可能的故障模式，并验证了当前实现的覆盖情况。

## 故障模式分类

### 1. 抓取阶段故障模式

| 故障模式 | 描述 | 检测方法 | 重试策略 |
|----------|------|----------|----------|
| grasp_failure | 物体没有移动 | check_grasp_failure | _retry_grasp |
| drop_failure | 物体 z 坐标下降 | check_drop_failure | _retry_drop |
| ik_failure | IK 求解失败 | check_ik_failure | _retry_ik |
| convergence_failure | PD 控制未收敛 | check_convergence_failure | _retry_convergence |
| path_planning_failure | 路径规划失败 | check_path_planning_failure | _retry_path_planning |
| weld_constraint_failure | 焊接约束添加失败 | check_weld_constraint_failure | _retry_weld |
| execution_exception | 执行异常 | try-except | _retry_exception |

### 2. 放置阶段故障模式

| 故障模式 | 描述 | 检测方法 | 重试策略 |
|----------|------|----------|----------|
| drop_failure | 物体掉落 | check_drop_failure | _retry_drop |
| position_drift | 物体偏离目标 | check_position_drift | _retry_drift |
| ik_failure | IK 求解失败 | check_ik_failure | _retry_ik |
| convergence_failure | PD 控制未收敛 | check_convergence_failure | _retry_convergence |
| weld_constraint_failure | 焊接约束删除失败 | check_weld_constraint_failure | _retry_weld |
| execution_exception | 执行异常 | try-except | _retry_exception |

### 3. 通用故障模式

| 故障模式 | 描述 | 检测方法 | 重试策略 |
|----------|------|----------|----------|
| action_failure | 动作执行失败 | execute_action 返回值 | _retry_generic |
| object_not_found | 物体不存在 | 物体名称检查 | 返回失败 |
| invalid_target | 目标位置无效 | 位置验证 | 返回失败 |

## 故障检测方法详解

### 1. check_grasp_failure
**检测逻辑**：比较抓取前后物体位置，如果移动距离 < 0.005m，认为抓取失败。
**适用场景**：pick 操作后检测物体是否被成功抓取。

### 2. check_drop_failure
**检测逻辑**：比较前后物体 z 坐标，如果下降 > 0.05m，认为物体掉落。
**适用场景**：lift 和 place 操作后检测物体是否掉落。

### 3. check_position_drift
**检测逻辑**：计算物体实际位置与目标位置的距离，如果 > 0.1m，认为位置漂移。
**适用场景**：place 操作后检测放置精度。

### 4. check_ik_failure
**检测逻辑**：计算 IK 解与目标位置的距离，如果 > 0.05m，认为 IK 失败。
**适用场景**：IK 求解后验证解的质量。

### 5. check_convergence_failure
**检测逻辑**：检查 PD 控制是否收敛到目标关节位置。
**适用场景**：PD 控制后验证收敛性。

### 6. check_path_planning_failure
**检测逻辑**：检查路径规划是否返回有效路径。
**适用场景**：路径规划后验证路径有效性。

### 7. check_weld_constraint_failure
**检测逻辑**：检查焊接约束操作是否成功。
**适用场景**：add_weld_constraint 和 delete_weld_constraint 后验证。

## 重试策略详解

### 1. _retry_grasp
**策略**：移动到安全高度 → 以 1cm 偏移重新接近 → 重新抓取
**适用**：抓取失败（物体没动）

### 2. _retry_drop
**策略**：移动到安全高度 → 重新抓取（带偏移）→ 重新放置
**适用**：物体掉落

### 3. _retry_drift
**策略**：移动到安全高度 → 以相反偏移重新抓取 → 精确放置
**适用**：放置位置不准确

### 4. _retry_ik
**策略**：移动到更高安全高度 → 以 2cm 偏移重试
**适用**：IK 求解失败

### 5. _retry_convergence
**策略**：等待更长时间 → 重试操作
**适用**：PD 控制未收敛

### 6. _retry_path_planning
**策略**：移动到更高安全高度 → 直接接近
**适用**：路径规划失败

### 7. _retry_weld
**策略**：移动到安全高度 → 等待物理稳定 → 重试操作
**适用**：焊接约束失败

### 8. _retry_exception
**策略**：移动到安全高度 → 等待恢复 → 重试操作
**适用**：执行过程中发生异常

### 9. _retry_generic
**策略**：直接重试原操作
**适用**：通用失败情况

## execute_with_recovery 增强功能

### 1. 异常捕获
```python
try:
    # 执行操作
except Exception as e:
    # 捕获异常并记录故障
```

### 2. 物体存在性检查
```python
if obj_name and obj_name not in scene_objects:
    return {"success": False, "reason": f"Object {obj_name} not found"}
```

### 3. 目标位置有效性检查
```python
if action_type == "place" and target_pos is not None:
    if len(target_pos) < 3 or target_pos[2] < 0:
        return {"success": False, "reason": "Invalid target position"}
```

### 4. 多故障模式检测
```python
if action_type == "pick":
    # 检查 grasp_failure 和 drop_failure
elif action_type == "place":
    # 检查 drop_failure 和 position_drift
```

## 故障模式覆盖矩阵

| 故障模式 | 检测 | 重试 | 覆盖状态 |
|----------|------|------|----------|
| grasp_failure | ✅ | ✅ | 完全覆盖 |
| drop_failure | ✅ | ✅ | 完全覆盖 |
| position_drift | ✅ | ✅ | 完全覆盖 |
| ik_failure | ✅ | ✅ | 完全覆盖 |
| convergence_failure | ✅ | ✅ | 完全覆盖 |
| path_planning_failure | ✅ | ✅ | 完全覆盖 |
| weld_constraint_failure | ✅ | ✅ | 完全覆盖 |
| execution_exception | ✅ | ✅ | 完全覆盖 |
| action_failure | ✅ | ✅ | 完全覆盖 |
| object_not_found | ✅ | - | 检测覆盖 |
| invalid_target | ✅ | - | 检测覆盖 |

## 重试机制

- **最大重试次数**：2 次（可配置）
- **中止条件**：连续 3 次失败
- **递归重试**：支持嵌套重试（重试计划中的动作也可能失败）

## 测试建议

1. **单元测试**：为每个故障检测方法编写测试
2. **集成测试**：测试完整的重试流程
3. **边界测试**：测试最大重试次数、中止条件
4. **异常测试**：测试各种异常情况

## 结论

当前实现已覆盖所有主要的故障模式，包括：
- 9 种故障检测方法
- 9 种重试策略
- 异常捕获和输入验证
- 多故障模式同时检测

故障模式覆盖完整，逻辑闭环实现可靠。
