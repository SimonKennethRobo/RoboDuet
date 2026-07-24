# RoboDuet 训练体系分析

## 一、总体架构：双智能体 + 单仿真环境

系统在同一个 IsaacGym 环境中同时训练两个独立的 PPO 智能体：

| 智能体 | 网络 | 控制自由度 | 输出维度 |
|---|---|---|---|
| **Dog** (`DogActorCritic`) | MLP + Adaptation Module | 四足 12 个腿关节 | 12 |
| **Arm** (`ArmActorCritic`) | MLP + Adaptation Module + History Encoder | 机械臂 6 关节 + 2 个"plan"输出 | 8 |

---

## 二、两阶段训练 Pipeline（GlobalSwitch 控制）

`go1_gym/utils/global_switch.py` 中的 `GlobalSwitch` 是核心开关，控制全局训练阶段切换。

### Stage 1（前 10000 iterations）：仅训练四足

- `global_switch.switch_open = False`
- 机械臂关节被强制锁死在默认位置（`legged_robot.py` `_keep_arm_fixed()`）
- Dog policy 学纯运动（速度跟踪、步态稳定）
- Arm policy 存在但不参与任何计算

### 过渡期

> **注意**：当前配置中过渡期长度为零。
>
> `scripts/auto_train.py` 第 106 行：
> ```python
> global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 0
> ```
>
> `GlobalSwitch` 里准备好的 sigmoid 曲线基础设施（`init_sigmoid_lr`、reward scale 插值）在当前配置下**完全没有被使用**。实际行为是到第 10000（或 2000）iteration 时 `open_switch()` 被调用，reward scales **瞬间**从 locomotion-only 切换到 hybrid 配置。过渡期是一个留了接口但目前跳过了的功能。

### Stage 2（切换后）：联合训练

- `global_switch.open_switch()` 触发
- Arm policy 开始输出动作，reward scales 切换为 hybrid 配置
- 两个 PPO 各自独立 update，共享同一批 rollout 数据

---

## 三、关键创新：`plan` 机制（arm → dog 的高层指令）

机械臂策略的输出维度为 8（`num_actions_arm_cd=8`）：

- **前 6 维**：直接控制机械臂 6 个关节（位置目标）
- **后 2 维（plan actions）**：输出机器人 body 的 pitch + roll 指令

```python
# VelocityTrackingEasyEnv.plan()
# 将 arm 的 plan_actions 写入 dog 的 commands
self.commands_dog[:, 3] = clip(rescaled_obs[..., 0], pitch_range)  # body pitch
self.commands_dog[:, 4] = clip(rescaled_obs[..., 1], roll_range)   # body roll
```

这形成了**隐式分层控制**：机械臂策略通过操作目标的几何关系，"告知"四足策略需要调整的躯干姿态，而不是直接输出腿部指令。

---

## 四、网络结构（CSE 风格，类 RMA）

两个 AC 都采用 **Asymmetric Actor-Critic + Adaptation Module**，推理时 actor 不需要特权信息。

### DogActorCritic（`go1_gym_learn/ppo_cse_automatic/dog_ac.py`）

```
obs_history (56 × 30 = 1680) ──→ adaptation_module [256, 128] ──→ latent (2 dims)
                                                                        │
actor:  cat(obs_history, latent)    ──→ [512, 256, 128] ──→ 12 leg actions
critic: cat(obs_history, privileged_obs) ──→ [512, 256, 128] ──→ value
```

privileged obs 只有 2 维（friction, restitution），极精简。

### ArmActorCritic（`go1_gym_learn/ppo_cse_automatic/arm_ac.py`）

结构更复杂，多了一个独立的 history encoder：

```
obs_history ──→ adaptation_module [256, 128] ──→ latent (9 dims)
                                                       │
obs_history[:-num_obs] ──→ actor_history_encoder [512, 256, 128] ──→ his_latent (128)
                                                                           │
actor:  cat(current_obs, latent, his_latent) ──→ [512, 256, 128] ──→ 8 actions
                                                                  (最后 2 个 plan actions 用 tanh 激活)
critic: cat(current_obs, privileged_obs, h_latent) ──→ value
```

Arm 的 privileged obs 包含末端执行器在机体坐标系下的 lpy + 四元数（9 维），信息量远大于 Dog。

---

## 五、对标准 PPO 的改动

| 改动点 | 标准 PPO | RoboDuet 改动 |
|---|---|---|
| **智能体数量** | 单 agent | **双 agent（alg_arm + alg_dog），共享环境** |
| **奖励流** | 单一 reward | **双 reward buffer**：`rew_buf_dog`（含步态奖励）、`rew_buf_arm`（排除速度跟踪奖励）|
| **优化器** | 一个 optimizer | **两个 optimizer**：PPO 主优化器 + adaptation_module 监督损失优化器 |
| **Adaptation Module** | 无 | **同步训练的有监督辅助损失**：MSE(pred\_latent, privileged\_obs)，80/20 train/test split |
| **Actor 输入** | obs | **obs\_history（30 帧堆叠）**，推理时无需特权信息 |
| **Critic 输入** | obs | **privileged\_obs（真实物理参数 + EE 位姿）** |
| **学习率调度** | 固定 or 衰减 | **Adaptive KL**：KL > 2×target → lr/1.5，KL < target/2 → lr×1.5 |
| **Timeout 处理** | 丢弃 | **Bootstrap**：`reward += γ × V(s) × timeout_mask` |
| **训练阶段** | 单阶段 | **两阶段 + GlobalSwitch**（当前过渡期为 0，瞬间切换）|
| **动作空间** | 单一 | **混合控制**：腿用力矩控制，臂用位置目标控制（control type "M"）|
| **plan actions** | 无 | **arm 输出的后 2 维 + tanh 激活**，写入 dog 的 body 姿态指令 |

---

## 六、观测空间设计

### Dog 观测（56 维，30 帧历史 = 1680 维输入）

```
gravity_vec(3) + leg_joint_pos(12) + leg_joint_vel(12) + leg_actions(12)
+ dog_commands(5: vx, vy, ω_z, pitch_cmd, roll_cmd)
+ arm_commands_or_zeros(6: 末端操作目标，switch 关闭时全零)
+ roll(1) + pitch(1)
```

arm 目标命令被包含在 dog 观测中，让狗知道末端执行器要去哪。

### Arm 观测（20 维，30 帧历史 = 600 维输入）

```
arm_joint_pos(6) + arm_actions(6)
+ arm_commands_lpy_rpy(6: 末端目标位置+姿态)
+ roll(1) + pitch(1)
```

### Privileged 观测

| | Dog | Arm |
|---|---|---|
| 内容 | friction(1), restitution(1) | friction(1), restitution(1), base\_mass(1), com\_displacement(3), ee\_lpy(3), ee\_quat\_in\_base(4) |
| 维度 | **2** | **9** (或更多，视配置) |

---

## 七、Arm Command 坐标系

### 坐标系定义：yaw-aligned base frame 球坐标

不是笛卡尔坐标，而是以机体为原点、沿 heading 对齐的**球坐标（lpy）**。

**坐标系特性**：
- 原点 = robot base position（随机器人平移）
- X 轴朝机器人当前偏航（heading）方向
- Z 轴垂直向上，z 值相对于地面高度（减去测量地形高度再加 0.38 m 机体高度偏置）
- **忽略 roll 和 pitch**，只用 yaw 对齐，即"水平投影帧"

### 球坐标分量（来自 `get_lpy_in_base_coord()`）

```python
l = sqrt(x² + y² + z²)         # 径向距离，范围 [0.3, 0.77] m
p = atan2(z, sqrt(x²+y²))      # 俯仰角（elevation），范围 ±0.45π ≈ ±81°
y = atan2(y, x)                 # 偏航角（azimuth），范围 ±π/2
```

### 末端姿态（另外 3 或 6 维）

- 默认：目标 roll/pitch/yaw（3 维欧拉角）
- 开启 `use_rot6d` 时：rot6d 表示（6 维），范围扩展
- 参考系：相对于 base yaw frame 的末端执行器姿态

### 直觉解释

> 给定的 command 是"末端执行器应该到机体前方多远、多高、偏多少角度的位置，以及在那里的姿态"。机器人转弯后这个 command 仍然有效，因为它跟着机体 heading 走，不随世界坐标系固定。

---

## 八、联合训练时两个 Policy 的独立性

### 梯度流完全独立

`alg_arm` 和 `alg_dog` 各自有独立的 rollout storage 和 optimizer，`update()` 各自调用，**没有任何共享 loss 或参数**。

### 通过环境产生隐性耦合（无梯度）

```
arm 输出 plan_actions
    → plan() 写入 commands_dog[:, 3/4]
    → dog 观测里包含这两个姿态命令
    → dog 被奖励按此姿态运动
    → dog 运动产生的 roll/pitch 进入 arm 的观测
    → arm 观测到的 ee 位置也被 dog 运动影响
```

这是典型的 Multi-Agent RL 中的 **cooperative but independent learners** 模式：两个 agent 通过环境状态互相影响，但各自更新自己的参数。

### 最终部署产物：两个独立文件组

```
deploy_model/
├── adaptation_module_latest_arm.jit
├── body_latest_arm.jit
├── history_latest_arm.jit      # arm 独有的 history encoder
├── adaptation_module_latest_dog.jit
└── body_latest_dog.jit
```
