# RoboDuet Stage-1 → rl_sar 部署分析

本文分两部分：

1. **robot_lab 训练的模型如何在 rl_sar 中做 sim2sim / sim2real** —— 摸清 rl_sar 的"合约"。
2. **RoboDuet stage-1 dog policy 如何导出成 rl_sar 兼容的模型+配置** —— 差异分析与改造方案。

所有结论均来自当前工作区代码（`refers/robot_lab/`、`rl_sar/`、`RoboDuet/`），行号引用可直接跳转。

---

## Part 1 — robot_lab 模型在 rl_sar 中的部署机制

### 1.1 rl_sar 的分层结构

```
                    ┌──────────────────────────────────────────┐
                    │  RL (rl_sdk.hpp/cpp)  —— 与机器人无关     │
                    │  ComputeObservation() / ComputeOutput()  │
                    │  ReadYaml() / InitRL() / FSM             │
                    └───────────▲──────────────────▲───────────┘
                                │ 纯虚               │
        ┌───────────────────────┴──┐      ┌─────────┴────────────────┐
        │ RL_Sim (sim2sim)         │      │ RL_Real  (sim2real)      │
        │  rl_sim_mujoco.cpp       │      │  rl_real_go2.cpp 等      │
        │  rl_sim.cpp (Gazebo)     │      │  unitree_sdk2 / DDS      │
        │  GetState/SetCommand     │      │  GetState/SetCommand     │
        └──────────────────────────┘      └──────────────────────────┘
```

关键点：**sim2sim 和 sim2real 共用同一套 `RL` 基类、同一份 policy 和同一份 yaml**。
两者唯一的区别是 `GetState()` / `SetCommand()` 这两个纯虚函数的实现，
以及 `ang_vel_axis`（ROS1 Gazebo 是 world 系，MuJoCo/ROS2/实物是 body 系，
见 `rl_sdk.cpp:77-87`）。这就是 rl_sar 的核心价值——**换仿真器/换实物零改动**。

### 1.2 robot_lab 侧导出什么

`refers/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py:203-206`：

```python
export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
```

导出的是 IsaacLab 的 `export_policy_as_jit`：一个 **无状态 TorchScript Module，
签名 `forward(obs: Tensor[1, N]) -> Tensor[1, A]`**，内部已把 normalizer 融进去。
只有 actor，没有 critic，没有 adaptation module。

rl_sar 侧 `InferenceRuntime::TorchModel::forward()`
（`rl_sar/src/rl_sar/library/core/inference_runtime/inference_runtime.cpp:58-88`）
就是把 `std::vector<float>` reshape 成 `{1, N}` 喂进去、取出输出。
`ModelFactory::load_model()` 按扩展名自动选 torch/onnx 后端。

> **合约 A：模型必须是单输入单输出的扁平 TorchScript / ONNX，输入 `[1, N]`，输出 `[1, A]`。**

### 1.3 两层 YAML 配置

```
policy/<ROBOT>/base.yaml            ← 机器人级，硬件顺序，构造时读一次
policy/<ROBOT>/<CONFIG>/config.yaml ← policy 级，进 RL 状态时读，覆盖 base
policy/<ROBOT>/<CONFIG>/policy.pt   ← 模型
```

加载顺序（`rl_sdk.cpp:469-488`，`ReadYaml` 逐 key 写入同一个 `params.config_node`）：

- `RL_Sim` 构造函数 → `ReadYaml(robot_name, "base.yaml")`（`rl_sim_mujoco.cpp:85`）
- 进入 `RLFSMStateRLLocomotion::Enter()` → `InitRL(robot_name + "/" + config_name)`
  → `ReadYaml(path, "config.yaml")`（`rl_sdk.cpp:229`）+ 加载模型（`rl_sdk.cpp:248-249`）

**同名 key 后读的覆盖先读的**，所以 `config.yaml` 可以改写 `base.yaml` 里的
`default_dof_pos` / `joint_mapping` / `torque_limits` 等。这就是"同一台机器人挂多个
policy、每个 policy 有自己的默认站姿"的实现方式。

### 1.4 一帧推理的完整数据流

以 MuJoCo sim2sim 为例（`rl_sim_mujoco.cpp`）：

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 1. 200 Hz 控制线程 | `loop_control`, `dt=0.005` | `GetState` → `StateController`(FSM) → `SetCommand` |
| 2. 50 Hz 推理线程 | `loop_rl`, `dt*decimation=0.02` | `RunModel()` |
| 3. 读传感器 | `GetState():158-172` | 用 `joint_mapping[i]` 从 `sensordata` 取第 i 个**策略序**关节 |
| 4. 填 obs 源 | `RunModel():335-344` | `ang_vel/commands/base_quat/dof_pos/dof_vel` |
| 5. 拼 obs | `ComputeObservation()` `rl_sdk.cpp:64-183` | 按 `observations` 列表逐项拼接 + `clip_obs` 截断 |
| 6. 历史缓冲 | `Forward():390-395` | `observations_history` 非空则走 `ObservationBuffer` |
| 7. 前向 | `model->forward({obs})` | + `clip_actions_lower/upper` 截断 |
| 8. 出力 | `ComputeOutput()` `rl_sdk.cpp:256-271` | `q = action*action_scale + default_dof_pos` |
| 9. 下发 | `RLControl()` `rl_sdk.cpp:589-609` | 队列取出 → `motor_command.q/kp/kd` |
| 10. 写执行器 | `SetCommand():176-188` | 用 `joint_mapping[i]` 写回**硬件序** |

`ComputeObservation()` 目前支持的**通用观测项**（`rl_sdk.cpp:68-112`）：

| 名称 | 维度 | 计算 |
|---|---|---|
| `lin_vel` | 3 | `obs.lin_vel * lin_vel_scale` |
| `ang_vel` | 3 | `obs.ang_vel * ang_vel_scale`（world 系则先 `QuatRotateInverse`） |
| `gravity_vec` | 3 | `QuatRotateInverse(base_quat, [0,0,-1])` |
| `commands` | 3 | `obs.commands * commands_scale` |
| `dof_pos` | N | `(dof_pos - default_dof_pos) * dof_pos_scale`，`wheel_indices` 置零 |
| `dof_vel` | N | `dof_vel * dof_vel_scale` |
| `actions` | N | 上一步动作原值 |

外加三个**任务专用项**（`rl_sdk.cpp:113-167`），以 `<项目名>/<项名>` 命名：
`whole_body_tracking/motion_command`、`whole_body_tracking/motion_anchor_ori_b`、
`RoboMimic_Deploy/phase`。**这就是官方给出的扩展模式** —— 后面 RoboDuet 照抄这个模式。

### 1.5 为什么 go2/robot_lab 能开箱即用

`refers/.../unitree_go2/rough_env_cfg.py:40-53` + `velocity_env_cfg.py:140-189` 的
observation group（`base_lin_vel=None`、`height_scan=None`）拼出来正好是：

```
base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
                + joint_pos(12) + joint_vel(12) + last_action(12) = 45
```

对应 `rl_sar/policy/go2/robot_lab/config.yaml`：

```yaml
num_observations: 45
observations: ["ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]
ang_vel_scale: 0.25   # ← observations.policy.base_ang_vel.scale
dof_pos_scale: 1.0    # ← joint_pos.scale
dof_vel_scale: 0.05   # ← joint_vel.scale
action_scale: [0.125, 0.25, 0.25, ...]   # ← actions.joint_pos.scale 的 dict 展开
clip_actions_lower/upper: ±100           # ← actions.joint_pos.clip
clip_obs: 100.0                          # ← ObsTerm.clip
```

**一一对应，逐项手抄**。robot_lab 没有自动导出 yaml 的脚本，这份 config.yaml
是人工从 env_cfg 翻译过来的。这一点对 Part 2 很重要——我们同样要自己写翻译器。

> **合约 B：观测项的名字、顺序、缩放、裁剪必须与训练环境逐项对齐；
> `joint_names`/`joint_controller_names`/`joint_mapping` 用硬件序，
> 其余所有向量（`default_dof_pos`/`rl_kp`/`action_scale`/`torque_limits`）用策略序。**

`joint_mapping` 的语义（三处实现完全一致）：

```cpp
// 训练序 i  →  硬件/仿真器索引 joint_mapping[i]
state->motor_state.q[i] = sensordata[joint_mapping[i]];                 // rl_sim_mujoco.cpp:169
state->motor_state.q[i] = low_state.motor_state()[joint_mapping[i]].q(); // rl_real_go2.cpp:159
mj_data->ctrl[joint_mapping[i]] = ...command.motor_command.q[i];         // rl_sim_mujoco.cpp:182
```

go2 是恒等映射（IsaacLab 里就按 `FR,FL,RR,RL` 定义 `joint_names`），
所以看不出来；`g1/whole_body_tracking/dance_102/config.yaml:27` 是非恒等的实例。

### 1.6 sim2sim → sim2real 需要改什么

| 项 | sim2sim (MuJoCo) | sim2real (Go2) |
|---|---|---|
| 入口 | `rl_sim_mujoco.cpp`，`./rl_sim_mujoco go2 scene` | `rl_real_go2.cpp`，`./rl_real_go2 <netif>` |
| 状态源 | `mj_data->sensordata` | `unitree_go::msg::dds_::LowState_` |
| 指令 | `mj_data->ctrl`（手算 PD 力矩） | `LowCmd_` 的 `q/dq/kp/kd/tau`（板载 PD） |
| `ang_vel_axis` | `"body"` | `"body"` |
| 资产 | `src/rl_sar_zoo/<ROBOT>_description/mjcf/<scene>.xml` | 不需要 |
| policy / yaml | **完全相同** | **完全相同** |
| FSM | 同一个 `fsm_<ROBOT>.hpp` | 同一个 |

所以在 rl_sar 里"跑通 sim2sim"基本等于"跑通 sim2real"，
剩下的是通信层和安全保护（`TorqueProtect` / `AttitudeProtect`）。

---

## Part 2 — RoboDuet Stage-1 导出到 rl_sar

### 2.1 Stage-1 dog policy 是什么

- 网络：`DogActorCritic.actor_body`，纯 `nn.Sequential`，
  `Linear(2460→512) ELU Linear(512→256) ELU Linear(256→128) ELU Linear(128→12)`
  （`go1_gym_learn/ppo_cse_automatic/dog_ac.py:62-76`，hidden dims `[512,256,128]`）
- `dog.use_adaptation_module = False`（`config/wbc.py:160`）
  → **actor 输入就是纯 obs history，没有 latent 拼接**，这对导出是极大利好。
- 输入：`dog_num_obs_history = 30 × 82 = 2460`
- 输出：12（`dog.num_actions_loco = 12`）
- **已经在训练时自动导出了 TorchScript**：
  `go1_gym_learn/ppo_cse_automatic/__init__.py:587-590`
  ```python
  body_dog_path = f"{path}/body_latest_dog.jit"
  body_model_dog = copy.deepcopy(self.alg_dog.actor_critic.actor_body).to("cpu")
  torch.jit.script(body_model_dog).save(body_dog_path)
  ```
  产物在 `<logdir>/deploy_model/body_latest_dog.jit`。

> **好消息：合约 A 天然满足。** `body_latest_dog.jit` 直接就是 rl_sar 要的
> 单输入单输出 TorchScript，改个名 `policy.pt` 丢进 `policy/go2_x5/roboduet/` 即可被
> `ModelFactory::load_model()` 加载。真正的工作全部在**观测拼装**这一侧。

### 2.2 82 维 dog observation 精确布局

来自 `go1_gym/envs/roboduet/wbc_env.py:2300-2485`（`get_dog_observations`）
与 `_dog_obs_layout():2235-2298`。缩放常数见 `config/legged_robot.py:360-383`。

| # | 偏移 | 维 | 内容 | 缩放 | rl_sar 现成？ |
|---|---|---|---|---|---|
| 1 | 0:3 | 3 | `projected_gravity` | 1.0 | ✅ `gravity_vec` |
| 2 | 3:15 | 12 | 腿 `dof_pos - default` | `dof_pos=1.0` | ⚠️ 需切片 |
| 3 | 15:27 | 12 | 腿 `dof_vel` | `dof_vel=0.05` | ⚠️ 需切片 |
| 4 | 27:39 | 12 | 腿 `actions`（上一步） | 1.0 | ⚠️ 需切片 |
| 5 | 39:45 | 6 | `commands_dog` | `[2, 2, 0.25, 1, 1, 1]` | ❌ rl_sar 只有 3 维 |
| 6 | 45:51 | 6 | `commands_arm_obs` | 1.0 | ❌ **stage-1 恒为 0** |
| 7 | 51:55 | 4 | `clock_inputs` | 1.0 | ❌ 需自己算步态时钟 |
| 8 | 55:58 | 3 | `base_ang_vel` | `ang_vel=0.25` | ✅ `ang_vel` |
| 9 | 58:61 | 3 | `base_lin_vel` | `lin_vel=2.0` | ⚠️ 实物不可观测 |
| 10 | 61:64 | 3 | `pose_actual = [height, pitch, roll]` | `[1, 1, 1]` | ⚠️ height 实物不可观测 |
| 11 | 64:67 | 3 | `pose_error = pose_target - pose_actual` | — | ❌ 派生量 |
| 12 | 67:70 | 3 | `velocity_error = cmd - actual` | — | ❌ 派生量 |
| 13 | 70:76 | 6 | 臂 `dof_pos - default` | `dof_pos=1.0` | ⚠️ 需切片 |
| 14 | 76:82 | 6 | 臂 `dof_vel` | `dof_vel=0.05` | ⚠️ 需切片 |

补充说明：

- **命令布局**（`wbc_env.py:27-43`）：
  `[x_vel, y_vel, yaw_vel, body_pitch, body_roll, body_height]`。
  注意 `pose_actual` 是 `[height, pitch, roll]`，**顺序和命令不一样**。
- `pose_target = [(base_height_target + cmd_height)*1.0, cmd_pitch*1.0, cmd_roll*1.0]`，
  `base_height_target = 0.34`（`config/go1.py`）。
- `velocity_error = [cmd_vx*2 - vx*2, cmd_vy*2 - vy*2, cmd_yaw*0.25 - wz*0.25]`。
- 每帧先按 `clip_observations = 100.0` 截断（`ObservationBuilder.build()`），
  **再** 入历史缓冲 —— 与 rl_sar `Forward()` 里"先 clamp 再 insert"的顺序一致。✅
- 历史布局（`wbc_env_wrapper.py:587-589`）：
  `cat((hist[:, 82:], obs), -1)` → **最旧在前、最新在后**。

### 2.3 关节顺序与控制参数

IsaacGym DOF 序 = URDF 树序（`resources/robots/go2_x5_v3/urdf/go2_x5.urdf`，只数
revolute/prismatic）：

```
0..2   FL_hip, FL_thigh, FL_calf
3..5   FR_hip, FR_thigh, FR_calf
6..8   RL_hip, RL_thigh, RL_calf
9..11  RR_hip, RR_thigh, RR_calf
12..17 x5_joint1 .. x5_joint6
18,19  x5_gripper_joint, x5_joint8   ← 夹爪，非策略关节
```

Unitree Go2 SDK 序是 `FR, FL, RR, RL`，所以：

```yaml
joint_mapping: [3,4,5, 0,1,2, 9,10,11, 6,7,8, 12,13,14,15,16,17]
#               FL      FR      RL       RR      arm (假定臂接在 12..17)
```

控制参数（`config/wbc.py` + `config/go1.py`）：

| 项 | 值 | 来源 |
|---|---|---|
| `dt` | 0.005 | `legged_robot.py:410` |
| `decimation` | 4 → 50 Hz 策略 | `go1.py:28` |
| `action_scale` | 0.25，hip × `hip_scale_reduction=0.5` | `go1.py:26-27` |
| 腿 kp / kd | 35.0 / 1.0 | `wbc.py:173-174` |
| 臂 kp | `[50, 50, 80, 30, 20, 20]` | `wbc.py:184-189` |
| 臂 kd | `[5, 10, 10, 2.5, 2, 1]` | `wbc.py:202-207` |
| `clip_actions` | 10.0 | `wtw.py:179` |
| `clip_observations` | 100.0 | `legged_robot.py:338` |
| 腿力矩上限 | hip/thigh 23.7，calf 45.43 | URDF `effort` |

默认关节角（策略序，`config/wbc.py:82-118`）：

```
FL: [ 0.1, 0.8, -1.5]    FR: [-0.1, 0.8, -1.5]
RL: [ 0.1, 1.0, -1.5]    RR: [-0.1, 1.0, -1.5]
arm x5_joint1..6: 全 0
```

注意 **RL/RR 的 thigh 是 1.0，FL/FR 是 0.8**，不是 go2 标准的全 0.8。

### 2.4 差异汇总

| # | 差异 | 影响 | 处理方式 |
|---|---|---|---|
| G1 | 观测里腿(12)与臂(6)不连续，中间夹着命令/时钟 | `dof_pos`/`dof_vel`/`actions` 三项不能直接用 | 新增带索引范围的观测项 |
| G2 | `commands` 是 6 维不是 3 维 | — | 新增 `roboduet/dog_commands` |
| G3 | `arm_commands` 6 维零填充槽 | stage-1 恒 0，但**必须占位** | 新增 `roboduet/arm_commands`（全 0） |
| G4 | `clock_inputs` 需要维护步态相位积分器 | rl_sar 无此状态 | 在 `RL` 里加 `gait_indices` |
| G5 | `pose_error` / `velocity_error` 是派生量 | — | 新增 `roboduet/tracking` |
| G6 | 动作 12 维但机器人 18 自由度 | `ComputeOutput` 尺寸不匹配、`RLControl` 越界 | 动作零填充到 18 |
| G7 | `base_lin_vel` + `body_height` 实物不可观测 | **sim2real 阻塞项** | 见 §2.7 |

### 2.5 改造方案

#### A. RoboDuet 侧：新增导出脚本 `scripts/export_rl_sar.py`

职责：从 `<logdir>` 读 `parameters.pkl` + `checkpoints_dog/ac_weights_*.pt`，
产出 rl_sar 目录树。**不要手抄常数** —— 全部从运行时 `cfg` 里取，这样以后改配置
不会静默漂移（robot_lab 那边就是手抄的，是已知的维护痛点）。

```
policy/go2_x5/base.yaml
policy/go2_x5/roboduet_stage1/config.yaml
policy/go2_x5/roboduet_stage1/policy.pt
```

模型导出直接复用现成逻辑（比 `body_latest_dog.jit` 更稳的是重新 script 一遍，
顺带做 dummy-input 数值校验）：

```python
from scripts.load_policy import _checkpoint_path, _load_run_parameters, \
                                _validate_checkpoint_layout, _model_args_from_checkpoint, \
                                _load_inference_state, _temporary_model_args
# ... 构造 DogActorCritic(use_adaptation_module=structure["uses_adaptation"]) ...
assert actor_critic.adaptation_module is None, \
    "rl_sar 单输入模型不支持 adaptation module；请用 dog.use_adaptation_module=False 训练"
ts = torch.jit.script(actor_critic.actor_body.cpu().eval())
ts.save(out / "policy.pt")
# 数值校验
x = torch.randn(1, cfg.dog.dog_num_obs_history)
assert torch.allclose(ts(x), actor_critic.actor_body(x), atol=1e-6)
```

yaml 生成（值全部来自 `cfg`）：

```yaml
# policy/go2_x5/base.yaml
go2_x5:
  dt: 0.005                 # cfg.sim.dt
  decimation: 4             # cfg.control.decimation
  num_of_dofs: 18           # num_actions_loco + num_actions_arm
  wheel_indices: []
  fixed_kp: [60.0 ×12, 50,50,80,30,20,20]   # 站起用的硬 PD，腿部可高于 rl_kp
  fixed_kd: [3.0 ×12,  5,10,10,2.5,2,1]
  torque_limits: [23.7,23.7,45.43, ×4, 27,27,27,7,7,7]
  default_dof_pos: [ 0.1,0.8,-1.5,  -0.1,0.8,-1.5,
                     0.1,1.0,-1.5,  -0.1,1.0,-1.5,
                     0.0,0.0,0.0,0.0,0.0,0.0]
  joint_names: [...]              # 硬件序：FR,FL,RR,RL + arm
  joint_controller_names: [...]   # 硬件序
  joint_mapping: [3,4,5, 0,1,2, 9,10,11, 6,7,8, 12,13,14,15,16,17]
```

```yaml
# policy/go2_x5/roboduet_stage1/config.yaml
go2_x5/roboduet_stage1:
  model_name: "policy.pt"
  num_observations: 82
  observations:
    - "gravity_vec"                    # 3
    - "roboduet/leg_dof_pos"           # 12
    - "roboduet/leg_dof_vel"           # 12
    - "roboduet/leg_actions"           # 12
    - "roboduet/dog_commands"          # 6
    - "roboduet/arm_commands"          # 6  (stage-1 全 0)
    - "roboduet/clock_inputs"          # 4
    - "ang_vel"                        # 3
    - "roboduet/base_lin_vel"          # 3
    - "roboduet/body_pose_actual"      # 3
    - "roboduet/body_pose_error"       # 3
    - "roboduet/velocity_error"        # 3
    - "roboduet/arm_dof_pos"           # 6
    - "roboduet/arm_dof_vel"           # 6
  observations_history: [29,28,27,...,2,1,0]   # 最旧在前，与 HistoryWrapper 一致
  observations_history_priority: "time"
  clip_obs: 100.0
  clip_actions_lower: [-10.0 ×18]
  clip_actions_upper: [ 10.0 ×18]
  num_of_dofs: 18
  num_policy_actions: 12                        # ← 新增 key，见 §B
  action_scale: [0.125,0.25,0.25, ×4,  0.0 ×6]
  rl_kp: [35.0 ×12, 50,50,80,30,20,20]
  rl_kd: [ 1.0 ×12,  5,10,10,2.5,2,1]
  ang_vel_scale: 0.25
  lin_vel_scale: 2.0
  dof_pos_scale: 1.0
  dof_vel_scale: 0.05
  # RoboDuet 专用
  roboduet:
    num_leg_dofs: 12
    num_arm_dofs: 6
    dog_commands_scale: [2.0, 2.0, 0.25, 1.0, 1.0, 1.0]
    arm_num_commands: 6
    base_height_target: 0.34
    gait_frequency: 3.0        # use_dynamic_gait=False → 固定
    gait_duration: 0.5
    gait_phases: [0.5, 0.0, 0.0]   # trotting: phases/offsets/bounds
    observe_lin_vel: true          # 由 cfg.dog.observe_lin_vel 决定是否零填充
    observe_pose_actual: true
    observe_track_error: true
  default_dof_pos: [...]  # 同 base.yaml
  joint_mapping: [...]    # 同 base.yaml
```

#### B. rl_sar 侧：`rl_sdk.cpp` 新增观测项

照 `whole_body_tracking/*` 的模式，在 `ComputeObservation()` 的
`// ============= Other Observations =============` 段（`rl_sdk.cpp:113` 之后）追加。
需要在 `RL` 类里加三个成员：`gait_indices`（float）、`obs.dog_commands`（6 维）、
`obs.leg_actions`（12 维，即上一步策略输出未填充版）。

核心几项（伪代码，风格与现有分支一致）：

```cpp
else if (observation == "roboduet/leg_dof_pos")
{
    int n = this->params.Get<int>("roboduet/num_leg_dofs");
    std::vector<float> v(this->obs.dof_pos.begin(), this->obs.dof_pos.begin() + n);
    std::vector<float> d = this->params.Get<std::vector<float>>("default_dof_pos");
    v = v - std::vector<float>(d.begin(), d.begin() + n);
    obs_list.push_back(v * this->params.Get<float>("dof_pos_scale"));
}
else if (observation == "roboduet/dog_commands")
{
    // [x_vel, y_vel, yaw_vel, body_pitch, body_roll, body_height]
    std::vector<float> c = {this->control.x, this->control.y, this->control.yaw,
                            this->control.body_pitch, this->control.body_roll,
                            this->control.body_height};
    obs_list.push_back(c * this->params.Get<std::vector<float>>("roboduet/dog_commands_scale"));
}
else if (observation == "roboduet/arm_commands")
{
    obs_list.push_back(std::vector<float>(this->params.Get<int>("roboduet/arm_num_commands"), 0.0f));
}
else if (observation == "roboduet/clock_inputs")
{
    // 与 LeggedRobot._step_contact_targets 逐行对齐（legged_robot.py:2711-2760）
    float freq = this->params.Get<float>("roboduet/gait_frequency");   // 3.0
    float dur  = this->params.Get<float>("roboduet/gait_duration");    // 0.5
    float dt   = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    this->gait_indices = std::fmod(this->gait_indices + dt * freq, 1.0f);
    // trotting: phases=0.5, offsets=0, bounds=0
    // foot_indices = [g+0.5, g+0, g+0, g+0.5]  (FL, FR, RL, RR)
    std::vector<float> fi = {this->gait_indices + 0.5f, this->gait_indices,
                             this->gait_indices,       this->gait_indices + 0.5f};
    bool standing = std::sqrt(x*x + y*y + yaw*yaw) < 0.1f;   // 与训练一致的 stand 判据
    std::vector<float> clock(4);
    for (int i = 0; i < 4; ++i)
    {
        float idx = standing ? 0.25f : std::fmod(fi[i], 1.0f);
        idx = (idx < dur) ? idx * (0.5f / dur)
                          : 0.5f + (idx - dur) * (0.5f / (1.0f - dur));
        clock[i] = std::sin(2.0f * M_PI * idx);
    }
    obs_list.push_back(clock);
}
```

`roboduet/body_pose_actual` / `body_pose_error` / `velocity_error` 同理，
分别读 `obs.base_height`（新增）、`QuaternionToEuler(base_quat)` 和 `obs.lin_vel`；
当对应的 `observe_*` 开关为 false 时直接推零向量（与训练侧行为完全一致）。

> **关键：`gait_indices` 必须在 `RLFSMStateRLLocomotion::Enter()` 里清零**，
> 和训练侧 reset 时 `gait_indices=0` 对齐；否则每次重进 RL 状态相位随机。

#### C. rl_sar 侧：动作零填充（G6）

`ComputeOutput()`（`rl_sdk.cpp:256-271`）假定 `actions.size() == num_of_dofs`。
RoboDuet 输出 12、机器人 18。`vector_math.hpp` 的逐元素算子会按
`min(size1, size2)` 截断（`vector_math.hpp:175-184`），结果 `output_dof_pos` 只有 12 个，
而 `RLFSMState::RLControl()` 会按 `num_of_dofs=18` 索引 → **越界读**。

最小改动：在 `Forward()` 返回后、写回 `obs.actions` 之前把动作补零到 `num_of_dofs`：

```cpp
// RL_Sim::RunModel()
this->obs.leg_actions = this->Forward();                 // 12 维，进观测用这个
this->obs.actions = this->obs.leg_actions;
this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);   // 补 6 个 0
```

补零后：臂的 `action_scale` 配成 0 → `output_dof_pos[12..17] = default_dof_pos[12..17]`，
臂用自己的 `rl_kp/rl_kd` 位置保持在默认姿态。这与训练侧
`_apply_stage1_arm_curriculum_actions()` 在 `intensity=0` 时的行为一致
（`wbc_env.py:574-582`，动作使臂目标 = 固定位姿）。

> 观测里的 `roboduet/leg_actions` 用**未填充的 12 维**，
> 不能用补零后的 18 维——这是最容易踩的坑。

#### D. FSM：`src/rl_sar/fsm_robot/fsm_go2_x5.hpp`

直接拷 `fsm_go2.hpp` 改：

- `rl.config_name = "roboduet_stage1"`（`fsm_go2.hpp:165`）
- `pre_running_pos` 扩到 18 维（后 6 个填 0）
- `RLFSMStateRLLocomotion::Enter()` 里加 `rl.gait_indices = 0.0f;`
- `GetType()` 返回 `"go2_x5"`，`REGISTER_FSM_FACTORY(Go2X5FSMFactory, "RLFSMStatePassive")`
- 在 `fsm_all.hpp` 加 `#include "fsm_go2_x5.hpp"`

#### E. 机器人描述：`src/rl_sar_zoo/go2_x5_description/`

MuJoCo sim2sim 需要 `mjcf/<scene>.xml`（`rl_sim_mujoco.cpp:64`）。
`GetState()` 假定 sensordata 布局为：

```
[0 .. N-1]        jointpos  ×18
[N .. 2N-1]       jointvel  ×18
[2N .. 3N-1]      jointactuatorfrc ×18
[3N .. 3N+3]      framequat (w,x,y,z)
[3N+4 .. 3N+6]    gyro
```

工作区已有 `go2_x5_description/urdf/go2_x5.urdf`，可用 MuJoCo 的
`compile` 或 robot_lab 的 `scripts/tools/convert_urdf.py` 转，然后**手工加传感器块**
并保证顺序按硬件序（FR,FL,RR,RL + arm）排列。

因为 stage-1 观测还要 `base_lin_vel` 和 `base_height`，MJCF 里还需额外加：

```xml
<framelinvel objtype="site" objname="imu"/>   <!-- 或 velocimeter，注意 world/body 系 -->
<framepos    objtype="site" objname="imu"/>
```

并在 `RL_Sim::GetState()` 里读出来填 `obs.lin_vel` / `obs.base_height`。
**注意**：训练侧 `base_lin_vel` 是**机体系**（`quat_rotate_inverse` 后的），
`framelinvel` 给的是世界系，需要自己转；`base_pos[:,2]` 是世界系 z（绝对高度）。

### 2.6 工作量清单

| 文件 | 动作 | 规模 |
|---|---|---|
| `RoboDuet/scripts/export_rl_sar.py` | 新增 | ~250 行 |
| `rl_sar/.../rl_sdk.hpp` | 加 `gait_indices` / `obs.base_height` / `obs.leg_actions` / `control.body_*` | ~15 行 |
| `rl_sar/.../rl_sdk.cpp` | `ComputeObservation()` 新增 ~10 个 `roboduet/*` 分支 | ~150 行 |
| `rl_sar/.../rl_sim_mujoco.cpp` | `GetState` 加 lin_vel/height；`RunModel` 动作补零 | ~20 行 |
| `rl_sar/.../fsm_robot/fsm_go2_x5.hpp` | 新增（拷 go2） | ~250 行 |
| `rl_sar/.../fsm_robot/fsm_all.hpp` | 加一行 include | 1 行 |
| `rl_sar/src/rl_sar_zoo/go2_x5_description/` | 新增 MJCF + 配置 | 中等 |
| `rl_sar/policy/go2_x5/**` | 由导出脚本生成 | 0（自动） |

### 2.7 风险与建议（重要）

**R1 — `base_lin_vel` / `body_height` 在实物上不可观测（G7）。**
当前 `dog.observe_lin_vel/observe_pose_actual/observe_track_error` 全为 `True`
（`config/wbc.py:162-164`），意味着 obs 的第 58:70 共 12 维里塞了机体线速度、
绝对机身高度以及它们的跟踪误差。Go2 低层 SDK 只给 IMU 加速度，没有可靠的
线速度/高度估计。**sim2sim 能跑（MuJoCo 有真值），sim2real 会直接失效。**

好消息是 RoboDuet 的设计已经考虑了这点：这三个开关**只控制零填充，不改变 obs 宽度**
（`wbc_env.py:2393-2429` 的注释明确写了 "Fixed width regardless of..."）。所以：

> **建议：面向部署的 stage-1 训练，用
> `dog.observe_lin_vel=False`、`dog.observe_pose_actual=False`、
> `dog.observe_track_error=False` 重训（或微调）。**
> obs 仍是 82 维，第 58:70 恒为 0，导出的 yaml 里对应项直接推零向量，
> sim2sim 与 sim2real 行为完全一致，无需状态估计器。

如果一定要用现有的全开策略，就只做 sim2sim 验证，或接一个外部
里程计/状态估计器（工作区里的 `ocs2_ros2` / `go2_x5_ocs2` 可能有现成的）。

**R2 — 观测噪声。** `dog.add_obs_noise = False`（`config/wbc.py:161`），
所以 stage-1 dog policy 训练时**没有观测噪声**。真机上传感器噪声会直接暴露。
建议部署前先把 `add_obs_noise` 打开重训一轮。

**R3 — `domain_rand.dog_obs_frame_drop_prob = 0.0`。** 同理，没有丢帧鲁棒性训练。

**R4 — 臂的存在改变了动力学。** stage-1 训练时臂被 `stage1_arm` 曲线扰动
（质量 0.1~2.0 倍缩放、EE 载荷 0~1.5 kg）。部署时臂如果被另一个控制器（比如 OCS2 侧）
驱动，其 dof_pos/dof_vel 会实时进入 dog obs —— 这是 stage-1 的设计意图，没问题；
但**臂的关节读数必须真实接入**，不能像 `arm_commands` 那样零填充。

**R5 — `clock_inputs` 相位漂移。** rl_sar 的 `loop_rl` 是软实时线程，
若发生调度抖动，`gait_indices` 的积分步长与训练时的固定 `dt=0.02` 会偏差。
建议按实际经过时间积分而非固定 `dt`，或至少监控 `loop_rl` 的实际周期。

**R6 — `num_of_dofs=18` 影响 FSM 全流程。** `Interpolate()`、`TorqueProtect()`、
CSV logger 都按 `num_of_dofs` 遍历，所以 `base.yaml` 里所有 18 维数组
（`fixed_kp/fixed_kd/torque_limits/default_dof_pos`）都必须补全，
漏一个就是静默的越界或错误行为。

### 2.8 分阶段验证计划

1. **离线数值对齐（最重要，先做）**：写一个 Python 脚本，从 IsaacGym 里 dump
   连续 N 帧的 `dog_obs_history` 与对应 actions；再用 C++ 侧（或先用 Python 复刻的
   rl_sar 观测拼装逻辑）喂同样的原始传感器量，**逐维比对 82 维观测**。
   任何一维不匹配都必须先解决——这一步能省掉 90% 的调试时间。
2. **模型一致性**：`policy.pt` 用同一批 `obs_history` 在 Python 和 rl_sar 的
   `test_inference_runtime` 里分别前向，比对 12 维输出，容差 1e-5。
3. **MuJoCo sim2sim**：`./rl_sim_mujoco go2_x5 scene`，先只做站立（不给速度指令），
   看 `clock_inputs` 是否进 stand 相位（全部 0.25 → `sin(π/2)=1`）、姿态是否稳定。
4. **加指令行走**，对比 IsaacGym play 的步态频率/足端相位。
5. **实物**：先 `RLFSMStatePassive` → `GetUp`，用 `fixed_kp/kd` 确认关节映射正确
   （看机器人是否摆出正确站姿），再切 RL 状态。打开 `TorqueProtect` /
   `AttitudeProtect`（`rl_sim_mujoco.cpp:362-363` 目前是注释掉的）。

---

## 附：一句话总结

rl_sar 的合约是"**扁平 TorchScript + 逐项对齐的 yaml + `joint_mapping` 桥接策略序与硬件序**"。
RoboDuet stage-1 的模型侧天然满足（`body_latest_dog.jit` 就是现成的、且无 adaptation module），
**全部工作量在观测拼装**：82 维里有 34 维是 rl_sar 现有通用项覆盖不到的
（6 维 dog 命令、6 维臂命令占位、4 维步态时钟、12 维位姿/速度跟踪、以及腿/臂不连续的切片），
按 rl_sar 已有的 `whole_body_tracking/*` 扩展模式新增一组 `roboduet/*` 观测项即可。
另外必须处理"12 维动作 vs 18 自由度"的补零问题。
**部署前的头号决策是 R1**：建议用 `observe_lin_vel/pose_actual/track_error = False` 重训，
这样不改 obs 宽度就能去掉对状态估计器的依赖，sim2real 才有可行性。
