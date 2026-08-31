# 论文项目文档：面向持续移动操作的 Policy-Aware RL–MPC

## 1. 研究任务

研究四足机械臂在**持续行走过程中**跟踪世界坐标系下给定的末端执行器 $SE(3)$ 轨迹。

目标不是原地或小范围移动 base 来扩大机械臂工作空间，而是机器人在连续 locomotion 状态下，同时实现：

- 四足底盘稳定、鲁棒地行走；
- 机械臂高精度跟踪 EE position + orientation；
- 抑制步态引起的 body 抖动向 EE 的传播；
- 合理分配 base 与 arm 的运动。

典型任务包括长距离扫描、沿墙轨迹跟踪、移动检测、持续端持物和 rough-terrain manipulation。

------

## 2. 核心问题

Locomotion 和 manipulation 对控制器的要求存在明显差异。

四足 locomotion 面临复杂接触、地形变化、打滑和模型不确定性，RL 已表现出很强的鲁棒性。机械臂精确操作则高度依赖运动学关系、关节约束、碰撞约束以及未来轨迹预测，这些结构更适合由 MPC 显式处理。

因此采用解耦架构：

**RL locomotion + MPC manipulation**

但解耦以后产生新的关键问题：

> MPC 所规划的 floating base 并不是由一个已知刚体模型直接控制，而是由一个 RL locomotion policy 闭环控制。

从 MPC 角度，真实系统是：

**locomotion command → RL policy + quadruped → actual base SE(3)**

而不是：

**locomotion command = actual base motion**

普通 robust RL policy 虽然能稳定行走，但其 command transient、gait oscillation 和不同 domain 下的响应可能并不一致，因此对 MPC 来说仍然是一个难以预测的黑盒。

本文的核心观点是：

> **A robust locomotion policy is not necessarily a good subsystem for predictive manipulation; it must also be predictable to the planner.**

即：

> **Robustness makes locomotion executable; predictability makes it plannable.**

------

## 3. 核心思路

整个方法按照：

**Train it to be predictable → Model what remains → Plan through that model**

三个阶段展开。

### 3.1 Response-Consistent Locomotion Policy

首先不直接对任意训练好的 RL policy 建复杂模型，而是在 RL 训练阶段主动塑造其宏观闭环响应。

在成熟 locomotion policy 的 velocity / gait / posture command parameterization 基础上，引入 **response consistency objective**。

对于相同 locomotion command，希望 policy 在不同：

- terrain；
- friction；
- robot dynamics；
- payload；
- arm configuration / motion；

条件下，不仅能够稳定完成任务，而且表现出尽可能一致的宏观 base response。

训练目标包括两部分：

**Robustness objective**

保证复杂地形、扰动和动力学随机化下的稳定 locomotion。

**Response consistency objective**

约束高层可见的：

- velocity settling behavior；
- overshoot；
- base angular response；
- body-height response；
- gait-induced oscillation statistics；

使其具有低复杂度、可重复的闭环动力学。

Response consistency 是软目标而不是硬约束。受到极端扰动时，policy 可以优先恢复稳定，避免为了可预测性牺牲 RL 的鲁棒性。

目标是得到：

> **predictably robust locomotion**

而不是单纯 robust locomotion。

------

## 4. Policy Closed-Loop Response Model

训练得到 response-consistent policy 后，对：

**RL policy + quadruped**

这一整体闭环系统进行辨识。

这里辨识的不是 leg torque level rigid-body dynamics，而是 MPC 真正看到的：

**locomotion command → base SE(3) response**

模型分为两层。

### 4.1 Nominal Low-Order Response

描述 locomotion command 变化引起的低频 base motion，例如：

- forward / lateral velocity transient；
- yaw response；
- body posture response。

优先采用简单的一阶或二阶模型，使 MPC 获得结构明确、计算高效的 nominal dynamics。

### 4.2 Gait-Phase-Dependent Residual

持续 locomotion 中，base 并不是平滑移动的平台。

即使平均速度完全正确，trot 仍会产生周期性的：

- $z$ oscillation；
- roll / pitch oscillation；
- body angular velocity；
- gait-phase-dependent transient。

因此进一步学习：

> **nominal closed-loop response + gait-phase-dependent residual**

而不是只预测平均 base trajectory。

如果实验表明 arm motion 对 base response 有明显影响，可以进一步将：

- arm configuration；
- arm velocity；
- short-horizon planned arm motion；

作为 model conditioning input，而不必第一版就显式计算 reaction wrench。

最终模型用于预测未来完整 floating-base $SE(3)$ trajectory。

------

## 5. Policy-Conditioned Dynamic Feasibility

仅知道“policy 会怎么响应”还不够，MPC 还需要知道：

> 哪些 base motion 是当前 locomotion policy 不应该被要求执行的。

因此建立一个轻量的 dynamic feasibility envelope。

它不是简单固定：

- roll < 某个角度；
- pitch < 某个角度。

而是考虑：

- locomotion speed；
- angular velocity；
- gait phase；
- terrain condition；
- arm / payload condition；

之后得到 policy 可稳定实现的动态姿态区域。

Response model 回答：

> **If I issue this command, what will happen?**

Feasibility model 回答：

> **Should I ask the locomotion policy to do this at all?**

二者共同让 RL-controlled base 对 MPC 变得透明。

------

## 6. Policy-Aware Floating-Base SE(3) MPC

给定 world-frame EE reference：

$T^{ref}_{WE}(t)$

MPC 联合优化：

- locomotion command trajectory；
- arm joint trajectory / acceleration。

关键区别在于：

> **base SE(3) trajectory 不是一个可以任意优化的自由轨迹。**

MPC 中的 floating-base state 必须满足前面辨识得到的 RL closed-loop dynamics。

即：

**MPC locomotion command**

→ **identified RL response model**

→ **predicted base SE(3)**

同时：

**arm trajectory**

→ **arm FK**

最后：

**predicted base SE(3) + arm motion**

→ **world-frame EE SE(3)**

因此 MPC 实际是在一个 learned locomotion controller 的闭环动态之上进行 trajectory optimization。

------

## 7. MPC 优化目标

MPC 主要考虑四类目标。

### EE SE(3) Tracking

最小化 world-frame EE 的：

- position error；
- orientation error。

### EE Stabilization During Locomotion

不仅最小化平均 pose error，还抑制：

- EE velocity error；
- acceleration；
- gait-induced jitter。

从而显式解决持续行走时 body oscillation 向 EE 的传播问题。

### Base Motion Quality

轻度约束：

- excessive roll / pitch；
- angular velocity；
- violent acceleration。

但不强制 base 永远水平，使 MPC 在必要时仍可以利用 body lean / height adjustment 帮助 manipulation。

### Arm Motion Quality

考虑：

- joint position / velocity / acceleration limit；
- actuator limit；
- singularity / manipulability；
- collision；
- motion smoothness。

------

## 8. MPC 核心约束

相比普通 base–arm trajectory optimization，本文最关键的是两类约束。

**RL closed-loop dynamics constraint**

规划出来的 base motion 必须符合真实 locomotion policy 的动态响应，而不是假设 base 可以精确跟踪任意轨迹。

**Policy-conditioned dynamic feasibility constraint**

规划结果必须位于 locomotion policy 当前能够稳定实现的动态区域。

因此该优化可以概括为：

> **Policy-aware base–arm trajectory optimization**

而不是普通的 floating-base trajectory optimization。

------

## 9. 三个主要创新点

### Contribution 1 — MPC-Compatible Locomotion Policy

提出 response-consistent RL training，在保持复杂地形和外部扰动鲁棒性的同时，使 locomotion policy 在不同动力学和负载条件下表现出更加一致、低复杂度的宏观闭环响应。

核心思想：

> **Shape the plant before modeling it.**

### Contribution 2 — Policy-Aware Response Model and Feasibility Envelope

辨识持续 locomotion 中 RL-controlled floating base 的完整 $SE(3)$ closed-loop dynamics，包括 command transient 与 gait-phase-dependent body oscillation，并进一步建立 state-dependent dynamic feasibility envelope。

核心思想：

> **Expose how the learned controller responds and what it can safely realize.**

### Contribution 3 — Policy-Constrained Floating-Base SE(3) MPC

给定 world-frame EE $SE(3)$ trajectory，联合优化 locomotion commands 与 arm trajectory，通过 learned policy response model 预测完整 base motion，并在 policy feasibility constraints 下实现持续移动中的精准 EE tracking。

核心思想：

> **Plan through the learned controller rather than around it.**

------

## 10. 关键实验

首先验证 **policy response consistency**：普通 robust RL 与本文 policy 在不同 terrain、friction、payload 和 dynamics randomization 下执行相同 command，比较 settling time variance、overshoot variance、body response dispersion 以及 disturbance robustness。

其次验证 **model predictability**：分别用 ideal model、一阶模型、辨识模型以及 gait-aware model 预测未来 0.2 / 0.5 / 1.0 s 的 base $SE(3)$，比较 prediction error，并验证 response-consistent policy 是否能够被明显更简单的模型准确描述。

然后进行 **MPC ablation**：比较 ideal-base MPC、simple first-order MPC、identified response MPC 和完整 policy-aware MPC，评价 EE SE(3) RMSE、EE jitter、planned–executed base trajectory error 和任务成功率。

最终在真正的 **continuous loco-manipulation** 场景验证：机器人连续行走 5–10 m，同时执行世界坐标系中的精确 EE trajectory，并测试不同 walking speed、rough terrain 和 payload。重点报告：

**EE tracking error / EE jitter versus walking speed**

以证明该方法解决的是持续移动操作，而非 stance 或小范围 base reposition。

------

## 11. 项目边界

第一阶段不引入过多额外自由度。

暂不考虑：

- MPC 在线优化 locomotion behavior parameters；
- gait switching；
- full contact schedule optimization；
- 显式 arm reaction wrench dynamics；
- vision / VLA。

第一篇工作的重点保持在：

> **response-consistent RL → closed-loop identification → policy-aware SE(3) MPC**

把 planning–policy gap 讲透。

后续可以自然扩展到：

- continuously parameterized locomotion dynamics；
- MPC-selectable response bandwidth；
- gait-frequency optimization；
- reaction-wrench-aware planning；
- cross-manipulator generalization。

------

## 12. 一句话项目定位

> **We co-design a robust locomotion policy and a predictive manipulation controller by shaping the RL-controlled base into a predictable closed-loop system, identifying its full SE(3) response during continuous locomotion, and explicitly planning through these learned dynamics for precise mobile manipulation.**