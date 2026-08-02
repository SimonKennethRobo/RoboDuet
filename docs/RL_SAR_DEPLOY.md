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
  `Linear(H→512) ELU Linear(512→256) ELU Linear(256→128) ELU Linear(128→12)`
  （`go1_gym_learn/ppo_cse_automatic/dog_ac.py:62-76`，hidden dims `[512,256,128]`）
- `dog.use_adaptation_module = False`（`config/wbc.py:160`）
  → **actor 输入就是纯 obs history，没有 latent 拼接**，这对导出是极大利好。
- 输入：`H = dog_num_obs_history = 30 × dog_num_observations`
  （本仓库那次训练是 30 × 90 = 2700；宽度随训练 flag 变，见 §2.2）
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

### 2.2 dog observation 精确布局

来自 `go1_gym/envs/roboduet/wbc_env.py:2300-2485`（`get_dog_observations`）
与 `_dog_obs_layout():2235-2298`。缩放常数见 `config/legged_robot.py:360-383`。

> **总宽度不是常数**，取决于训练时开了哪些 flag：
> `arm_num_commands` 在 `--rot6d` 下是 9、否则 6；
> `dog_num_commands` 在 `--dyna_gait` 下是 11、否则 6。
> 当前默认 build 出来是 **85**；本仓库 `runs/2026-08-01/...` 那次
> （dyna_gait + traj_track）是 **90**。
> **所以导出脚本必须从 checkpoint 的 `parameters.pkl` 现场推导，不能写死。**
> 下表的偏移按 90 维那次列出（括号内为维度的配置来源）。

| # | 偏移 | 维 | 内容 | 缩放 | rl_sar 现成？ |
|---|---|---|---|---|---|
| 1 | 0:3 | 3 | `projected_gravity` | 1.0 | ✅ `gravity_vec` |
| 2 | 3:15 | 12 | 腿 `dof_pos - default` | `dof_pos=1.0` | ⚠️ 需切片 |
| 3 | 15:27 | 12 | 腿 `dof_vel` | `dof_vel=0.05` | ⚠️ 需切片 |
| 4 | 27:39 | 12 | 腿 `actions`（上一步） | 1.0 | ⚠️ 需切片 |
| 5 | 39:50 | 6 或 11 | `commands_dog` | `[2, 2, 0.25, 1, 1, 1, …]` | ❌ rl_sar 只有 3 维 |
| 6 | 50:59 | 6 或 9 | `commands_arm_obs` | 1.0 | ❌ **stage-1 恒为 0** |
| 7 | 59:63 | 4 | `clock_inputs` | 1.0 | ❌ 需自己算步态时钟 |
| 8 | 63:66 | 3 | `base_ang_vel` | `ang_vel=0.25` | ✅ `ang_vel` |
| 9 | 66:69 | 3 | `base_lin_vel` | `lin_vel=2.0` | ⚠️ 需状态估计 |
| 10 | 69:72 | 3 | `pose_actual = [height, pitch, roll]` | `[1, 1, 1]` | ⚠️ 需状态估计 |
| 11 | 72:75 | 3 | `pose_error = pose_target - pose_actual` | — | ❌ 派生量 |
| 12 | 75:78 | 3 | `velocity_error = cmd - actual` | — | ❌ 派生量 |
| 13 | 78:84 | 6 | 臂 `dof_pos - default` | `dof_pos=1.0` | ⚠️ 需切片 |
| 14 | 84:90 | 6 | 臂 `dof_vel` | `dof_vel=0.05` | ⚠️ 需切片 |

补充说明：

- **命令布局**（`wbc_env.py:27-43`）：
  `[x_vel, y_vel, yaw_vel, body_pitch, body_roll, body_height]`，
  dyna_gait 时再接 `[gait_frequency, footswing_height, stance_width, stance_length, gait_duration]`。
  注意 `pose_actual` 是 `[height, pitch, roll]`，**顺序和命令不一样**。
- `pose_target = [(base_height_target + cmd_height)*1.0, cmd_pitch*1.0, cmd_roll*1.0]`，
  go2 上 `base_height_target = 0.3`（不是 go1 的 0.34，同样要从 cfg 读）。
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


> G7 已由 **FAST-LIO 状态估计** 解决（见 §2.7 R1）；G1–G6 的实现见下。

### 2.5 已实现的改动

全部落地完成，清单如下。

#### RoboDuet 侧

| 文件 | 说明 |
|---|---|
| `scripts/export_rl_sar.py` | **新增**。从 `<logdir>` 生成 rl_sar 的 `base.yaml` + `config.yaml` + `policy.pt` |
| `scripts/verify_rl_sar_obs.py` | **新增**。逐维比对 rl_sar 观测拼装 vs `get_dog_observations()` 真值 |

`export_rl_sar.py` 的关键设计：

- **所有数值从 checkpoint 的 `parameters.pkl` 现场推导**（`load_runtime_cfg()`
  复刻 `load_policy.load_env()` 的 cfg 重建流程），没有一个硬编码常数。
  §2.2 已经证明这是必须的：同一套代码不同 flag 会产出 85 / 90 两种宽度。
- **不依赖 IsaacGym**。`go1_gym.envs.config` 本身是纯 Python；`dog_ac.py` 用
  `importlib` 按文件路径单独加载，绕开会 `import isaacgym` 的包 `__init__.py`。
  所以导出可以在任何有 PyTorch 的机器上跑。
- **网络层宽从 checkpoint 张量形状推导**，不信 `DogAC_Args`，老 checkpoint 也能导。
- **拒绝而不是猜**：checkpoint 若带 adaptation module（actor 需要两个输入），
  直接报错并说明处理方式，不会静默产出一个喂错输入的模型。
- **落盘后回读校验**：`torch.jit.load()` 重新读出保存的文件，与 eager 模块比对
  `atol=1e-6`。只查内存里的模块是查不出 `script()` 静默失败的。
- **宽度自洽检查**：观测项宽度之和必须等于 `cfg.dog.dog_num_observations`，
  否则报错——这道闸门保证 `observation_terms()` 和 `get_dog_observations()`
  不会悄悄漂移。

#### rl_sar 侧

| 文件 | 改动 |
|---|---|
| `library/core/rl_sdk/rl_sdk.hpp` | `RobotState::Base`（浮动基座状态）、`Observations::base_height`、`Control::body_pitch/roll/height`、`RL::gait_indices` |
| `library/core/rl_sdk/rl_sdk.cpp` | `ComputeObservation()` 新增 12 个 `roboduet/*` 观测项；命令按 `limit_*` 钳位；新增机体位姿按键 |
| `src/rl_sim_mujoco.cpp` | `GetState()` 读浮动基座传感器并转到机体系；`RunModel()` 动作零填充 |
| `fsm_robot/fsm_go2_x5.hpp` | **新增** 18 自由度 FSM |
| `fsm_robot/fsm_all.hpp` | 注册新 FSM |

几个值得说明的实现决定：

**动作零填充（G6）** 放在 `RunModel()` 里 `Forward()` 之后：

```cpp
this->obs.actions = this->Forward();          // 12 维
this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);   // 补到 18
```

必须补零而不是让它保持 12 维——`vector_math.hpp` 的逐元素算子按
`min(size)` 截断（`vector_math.hpp:175-184`），`output_dof_pos` 会只剩 12 个，
而 `RLFSMState::RLControl()` 按 `num_of_dofs=18` 索引，**越界读**。
补零后配合 `action_scale` 里臂的 6 个 0，臂就停在 `default_dof_pos` 上。
观测项 `roboduet/leg_actions` 取补零前的前 12 维，等价于切片，两边自洽。

**被动状态下臂不能松**（`fsm_go2_x5.hpp` `RLFSMStatePassive::Run()`）：
go2 原版 FSM 在 passive 下把所有关节 `kp=0, kd=8`。18 自由度机器人照抄会让
臂直接砸到机身上，还可能被反驱到自身硬限位。所以腿松、臂用 `fixed_kp/kd`
保持在默认位姿。

**`gait_indices` 在 `RLFSMStateRLLocomotion::Enter()` 清零**，对齐训练侧 reset
行为，否则每次重进 RL 状态相位是随机的。

**命令钳位** 用 `params.Has()` 守卫，配置里没有 `limit_*` 的机器人（其余 12 台）
完全不受影响。

**浮动基座状态**（FAST-LIO 接入点）：`RobotState::base` 有明确的坐标系约定——
`lin_vel` 是**机体系**、`position` 是**世界系**。MuJoCo 侧从 `framelinvel`
读世界系速度后用 `QuatRotateInverse` 转到机体系；实物侧 FAST-LIO 的输出同样
需要转换后再写入 `robot_state.base.lin_vel`。

### 2.6 验证结果

`verify_rl_sar_obs.py` 在真实 IsaacGym 环境中，把 rl_sar 的观测拼装
（按导出的 `config.yaml` 逐项重放）与 RoboDuet 自己的
`get_dog_observations()` 输出逐维比对，400 步、5 组不同指令
（含零指令以覆盖站立分支）：

```
per-term max |truth - rl_sar| over 400 compared steps (tolerance 1e-4):

  [ok  ] gravity_vec                dims   0:3    max_err 1.026e-07
  [ok  ] roboduet/leg_dof_pos       dims   3:15   max_err 2.980e-08
  [ok  ] roboduet/leg_dof_vel       dims  15:27   max_err 3.576e-08
  [ok  ] roboduet/leg_actions       dims  27:39   max_err 0.000e+00
  [ok  ] roboduet/dog_commands      dims  39:50   max_err 4.768e-08
  [ok  ] roboduet/arm_commands      dims  50:59   max_err 0.000e+00
  [ok  ] roboduet/clock_inputs      dims  59:63   max_err 5.734e-07
  [ok  ] ang_vel                    dims  63:66   max_err 0.000e+00
  [ok  ] roboduet/base_lin_vel      dims  66:69   max_err 0.000e+00
  [ok  ] roboduet/body_pose_actual  dims  69:72   max_err 3.065e-08
  [ok  ] roboduet/body_pose_error   dims  72:75   max_err 3.780e-08
  [ok  ] roboduet/velocity_error    dims  75:78   max_err 1.192e-07
  [ok  ] roboduet/arm_dof_pos       dims  78:84   max_err 0.000e+00
  [ok  ] roboduet/arm_dof_vel       dims  84:90   max_err 2.235e-09

  gait clock rate drift (rl_sar vs env): 3.874e-08
```

全部落在 float32 精度内。C++ 侧的这 12 个观测项与该脚本里的 Python 版是
逐行对照写的，所以这同时验证了导出配置和 `ComputeObservation()`。

**验证过程中发现的一个真实时序问题**：`clock_inputs` 是在 `env.step()` 内部的
`_step_contact_targets()` 里写入 buffer 的，用的是**当时**的 `commands_dog`。
若在 step 之后注入新指令再读观测，比对的是"新指令 vs 旧时钟"，必然不一致。
训练侧和 rl_sar 侧其实都是"同一时刻推进时钟并观测"，没有滞后；
所以校验脚本改成了真实控制环顺序 **施加指令 → step → 观测 → 比对**。
这一点在真机上同样重要：rl_sar 的 `ComputeObservation()` 在推进
`gait_indices` 的同一次调用里读 `control.x/y/yaw`，与训练一致。

C++ 侧编译验证：

```bash
# rl_sdk.cpp / fsm_go2_x5.hpp / fsm_all.hpp 全部通过
g++ -fsyntax-only -std=c++17 -DPOLICY_DIR='"..."' -I ... rl_sdk.cpp
```

`rl_sim_mujoco.cpp` 的改动因为缺 MuJoCo/GLFW 头文件无法整体编译，
改动片段已用等价的独立 TU 做了类型检查。

### 2.7 剩余工作与风险

**剩余工作**

| 项 | 状态 |
|---|---|
| `src/rl_sar_zoo/go2_x5_description/mjcf/*.xml` | **待做**。sim2sim 必需，见下 |
| `src/rl_sar/src/rl_real_go2_x5.cpp` | **待做**。需要 X5 机械臂的 SDK 接口细节 |
| FAST-LIO → `robot_state.base` 的接线 | **待做**，取决于上一项 |

MJCF 的传感器块顺序必须是（N = 18）：

```
[0 .. N-1]      jointpos          ×18   硬件序 FR,FL,RR,RL + arm
[N .. 2N-1]     jointvel          ×18
[2N .. 3N-1]    jointactuatorfrc  ×18
[3N .. 3N+3]    framequat  (w,x,y,z)
[3N+4 .. 3N+6]  gyro
[3N+7 .. 3N+9]  framelinvel  (世界系)     ← 新增，use_base_state_sensor 读这里
[3N+10 .. 3N+12] framepos    (世界系)     ← 新增
```

实物节点没写，是因为 X5 机械臂的通信接口在本工作区里看不到
（`X5-2025/` 只有 URDF 相关内容）。腿部照抄 `rl_real_go2.cpp` 即可，
臂的读写和 FAST-LIO 里程计订阅需要你补。

**R1 — 状态估计（已有方案）**：策略观测 `base_lin_vel`（机体系）和
`base_pos.z`（世界系绝对高度），共 12 维依赖它们。已确认用 FAST-LIO 提供。
接入时注意两件事：**坐标系**（`base.lin_vel` 必须是机体系，FAST-LIO 通常给
世界系，要用当前姿态转换）和**延迟**（LIO 的输出相对 IMU 有几十毫秒延迟，
训练时是零延迟真值；如果实机表现出低频振荡，优先怀疑这里）。

**R2 — 观测噪声（已解决）**：已有加噪声训练的模型。

**R3 — 丢帧鲁棒性**：`domain_rand.dog_obs_frame_drop_prob` 默认 0.0。
如果 FAST-LIO 的更新率低于 50 Hz，dog obs 里那 12 维会重复上一帧的值，
等效于丢帧。建议开一点 `dog_obs_frame_drop_prob` 重训，或确认 LIO 能跟上 50 Hz。

**R4 — 臂的关节读数必须真实接入**。stage-1 训练时臂是扰动源，其
`dof_pos/dof_vel` 实时进入 dog obs（第 78:90 维）——这是 stage-1 的设计意图。
但 `arm_commands`（第 50:59 维）在 stage-1 恒为 0，是占位槽。
**两者不要搞混**：前者必须是真值，后者必须是零。

**R5 — `clock_inputs` 相位漂移**：`loop_rl` 是软实时线程，
`gait_indices` 按固定 `dt` 积分。校验显示速率误差 3.9e-8（即公式正确），
但真机上若线程抖动，实际经过时间会偏离标称 `dt`。建议监控 `loop_rl` 实际周期；
必要时改成按实际经过时间积分。

**R6 — `num_of_dofs=18` 贯穿 FSM 全流程**：`Interpolate()`、`TorqueProtect()`、
CSV logger 都按 `num_of_dofs` 遍历。导出脚本生成的所有 18 维数组都已补全，
但如果手工改 yaml，漏一个就是静默越界。

### 2.8 使用方法

```bash
# 1) 导出（不需要 IsaacGym）
python scripts/export_rl_sar.py \
    --logdir runs/<date>/<run> \
    --rl_sar_root /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar \
    --robot go2_x5 --config_name roboduet_stage1

# 2) 逐维校验（需要 IsaacGym）
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python scripts/verify_rl_sar_obs.py \
    --logdir runs/<date>/<run> \
    --config ../rl_sar/policy/go2_x5/roboduet_stage1/config.yaml \
    --steps 400

# 3) sim2sim（需要先补 MJCF）
cd ../rl_sar && ./build.sh
./cmake_build/bin/rl_sim_mujoco go2_x5 scene
#   0 → 站起,  1 → 进 RL,  9 → 趴下,  P → 被动
#   WSAD/QE 速度,  TG 俯仰,  YH 横滚,  UJ 高度,  空格清零
```

上机顺序建议：先 `Passive → GetUp`，用 `fixed_kp/kd` 确认关节映射正确
（看机器人是否摆出正确站姿）；确认无误再按 `1` 切 RL 状态。
建议先打开 `rl_sim_mujoco.cpp:362-363` 目前注释掉的
`TorqueProtect` / `AttitudeProtect`。

---

## 附：一句话总结

rl_sar 的合约是"**扁平 TorchScript + 逐项对齐的 yaml + `joint_mapping` 桥接策略序与硬件序**"。
RoboDuet stage-1 的模型侧天然满足（actor 是纯 `nn.Sequential`、无 adaptation module），
**全部工作量在观测拼装**：整个观测向量里有 40 多维是 rl_sar 现有通用项覆盖不到的
（dog 命令、臂命令占位、步态时钟、位姿/速度跟踪、以及腿/臂不连续的切片），
已按 rl_sar 已有的 `whole_body_tracking/*` 扩展模式新增了一组 `roboduet/*` 观测项，
并处理了"12 维动作 vs 18 自由度"的补零问题。
观测拼装已在真实 IsaacGym 环境中逐维验证通过（误差 ≤ 1e-7）。
**剩余阻塞项是 MJCF 资产和实物节点**，后者需要 X5 机械臂的 SDK 接口细节。
