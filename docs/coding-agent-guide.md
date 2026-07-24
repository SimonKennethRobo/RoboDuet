# Coding Agent 执行文档：基于 RoboDuet 实现全身 SE(3) 轨迹跟踪

> **本文档面向 coding agent（如 Claude Code / Cursor）**。每个模块给出：要改什么文件、输入输出接口、核心算法伪代码、测试标准。按实现顺序排列，前面的模块是后面的依赖。
>
> **代码基础**：RoboDuet 的开源 codebase。RoboDuet 已有双 policy 架构（arm policy + leg policy）、Isaac Gym 训练环境、基本的 WTW 腿部 policy。我们在此基础上做修改和扩展。
>
> **硬件假设**：四足（Go2 级别）+ 6-DoF 或 7-DoF 机械臂。平地，自由空间。

---

## 0. 实现路线图与依赖关系

```
阶段 0: 基础设施（无 RL，纯计算模块）
  ├─ M1: 轨迹表示（γ(s) + s_ref(t)）
  ├─ M2: 可达性表（ρ 的 2D/4D 查表）
  ├─ M3: 在线 base_nom 计算
  ├─ M4: DLS-IK 求解器
  └─ M5: 轨迹生成器

阶段 1: 腿部 Policy（Phase 0）
  └─ M6: WTW 带臂重训

阶段 2: 上层 Policy（Phase 1）
  ├─ M7: 环境封装（Env wrapper）
  ├─ M8: 观测/动作/Reward 定义
  ├─ M9: Multi-critic PPO
  └─ M10: 课程管理器

阶段 3: 部署与评测
  ├─ M11: 安全滤波器（可选）
  └─ M12: 评测脚本
```

每个模块标注了依赖，严格按顺序实现。**不要跳步。**

---

## M1: 轨迹表示

### 位置

新建 `modules/trajectory.py`

### 要实现的类

```python
class Gamma:
    """弧长参数化的 SE(3) 几何路径"""

    def __init__(self, s_grid, p, R, tangent, lam=0.15):
        """
        s_grid: (N,)     弧长采样点
        p:      (N, 3)   位置
        R:      (N, 3, 3) 旋转矩阵
        tangent:(N, 3)   切线方向
        lam:    float    旋转-平移等价系数 (m/rad)
        """
        self.L = s_grid[-1]  # 总弧长

    def p_at(self, s):
        """给定弧长 s (batch,)，返回位置 (batch, 3)。线性插值。"""

    def R_at(self, s):
        """给定弧长 s (batch,)，返回旋转 (batch, 3, 3)。SLERP 插值。"""

    def tangent_at(self, s):
        """给定弧长 s (batch,)，返回切线 (batch, 3)。"""

    def rot_6d_at(self, s):
        """返回旋转的 6D 表示 (batch, 6)。取 R 的前两列并 reshape。"""


class TimeLaw:
    """时间律 s_ref(t)"""

    def __init__(self, t_grid, s_of_t, sdot_of_t):
        """
        t_grid:    (T,)  时间采样点
        s_of_t:    (T,)  每个时刻对应的弧长
        sdot_of_t: (T,)  每个时刻的弧长速度 ds/dt
        """

    def s_ref(self, t):
        """给定时间 t (batch,)，返回参考弧长 (batch,)。"""

    def sdot_ref(self, t):
        """给定时间 t (batch,)，返回参考弧长速度 (batch,)。"""


class TrajectoryBatch:
    """GPU 上 N 个环境各自持有一条 (gamma, time_law) 的 batched 容器"""

    def __init__(self, N, max_points, device):
        # 预分配 GPU 内存
        self.gamma_p = torch.zeros(N, max_points, 3, device=device)
        self.gamma_R = torch.zeros(N, max_points, 3, 3, device=device)
        # ... 其他字段

    def load(self, env_ids, gammas, time_laws):
        """把新生成的轨迹写入指定环境的 slot"""

    def sample_preview(self, s_current, L_h, K=9):
        """
        给定当前弧长 s_current (N,)，返回 K 个 preview 点。
        按路径距离采样，近密远疏。

        Returns:
            s_k:   (N, K)     preview 点的弧长
            p_k:   (N, K, 3)  preview 点的位置
            R_k:   (N, K, 3, 3)  preview 点的旋转
            sdot_k:(N, K)     preview 点的参考速度
        """
        Delta = L_h * torch.tensor(
            [0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.85, 1.0],
            device=self.device
        )  # (K,)
        s_k = (s_current.unsqueeze(1) + Delta.unsqueeze(0)).clamp(max=self.L)
        # ... 插值查询
```

### 关键函数

```python
def to_canonical(p_raw, R_raw, dt, lam=0.15, ds_grid=0.005):
    """
    把原始位姿序列转成 (Gamma, TimeLaw)。

    步骤：
    1. 低通滤波（Butterworth, fc ≈ 10Hz）
    2. 计算 SE(3) 弧长增量：ds² = ||dp||² + (λ·dθ)²
    3. 累积得 s_of_t → 这就是 TimeLaw
    4. 按 ds_grid 均匀重采样位置（PCHIP）和姿态（SLERP）→ Gamma
    """

def update_s(s_prev, ee_p, ee_R, gamma, window=0.15, M=16, lam=0.15):
    """
    前向窗口投影。返回 (s_new, d_lat)。
    全 batched：s_prev (N,), ee_p (N,3), ee_R (N,3,3)。

    注意：
    - 只向前搜索，s 单调不倒退
    - 用 SE(3) 距离（同一个 λ）
    - d_lat 用于终止条件和 reward
    """
```

### 测试标准

1. 构造一条已知的圆弧轨迹，验证 to_canonical 后 L ≈ 2πr
2. 验证 update_s 在自交路径上不跳（构造 8 字形）
3. 验证 sample_preview 在 s 接近 L 时 clamp 正确
4. GPU batch 性能：N=4096 时 sample_preview < 0.1ms

---

## M2: 可达性表

### 位置

新建 `modules/reachability.py`

### 要实现的内容

```python
class ReachabilityTable:
    """方向相关的可达半径表，支持 2D 和 4D"""

    def __init__(self, table, mode='2d'):
        """
        table: (n_az, n_el) for 2D
               (n_az_pos, n_el_pos, n_az_tool, n_el_tool) for 4D
        """
        self.table = table  # 常驻 GPU
        self.mode = mode

    @staticmethod
    def build_2d(fk_fn, q_lo, q_hi, jac_fn=None,
                 n_samples=2_000_000, n_az=72, n_el=36,
                 sigma_rot_min=0.05):
        """
        离线建表。

        步骤：
        1. 随机采 n_samples 组关节角，uniform(q_lo, q_hi)
        2. 排除自碰构型（调用碰撞检测）
        3. 正运动学 → 末端位置 p (M,3)
        4. 变换到肩坐标系
        5. [可选] 计算旋转子雅可比 J_rot = jac[:, 3:, :] 的最小奇异值
           排除 σ_min < sigma_rot_min 的构型
        6. 按方向 (az, el) 分桶
        7. 每桶取 r 的 0.98 分位数
        8. 高斯平滑 (sigma=1 grid cell)
        """

    @staticmethod
    def build_4d(fk_fn, q_lo, q_hi,
                 n_samples=40_000_000,
                 n_az_pos=72, n_el_pos=36,
                 n_az_tool=24, n_el_tool=12):
        """
        和 build_2d 相同流程，额外：
        - 正运动学同时输出末端姿态 R_ee
        - 提取工具指向 = R_ee 在肩坐标系下的第三列
        - 按 (az_pos, el_pos, az_tool, el_tool) 4D 分桶
        """

    def query(self, u, tool_dir=None):
        """
        查询 R_max。

        Args:
            u: (N, K, 3) 方向单位向量（肩坐标系）
            tool_dir: (N, K, 3) 工具指向（肩坐标系），仅 4D 时需要

        Returns:
            R_max: (N, K)
        """
        if self.mode == '2d':
            az = torch.atan2(u[..., 1], u[..., 0])
            el = torch.asin(u[..., 2].clamp(-1, 1))
            return bilinear_sample(self.table, az, el)
        else:
            # 4D: 多线性插值（16 角点加权平均）
            return multilinear_4d(self.table, az_pos, el_pos, az_tool, el_tool)


def compute_rho(p_tgt, T_base, T_mount, table, tool_dir_world=None):
    """
    计算 ρ = r / R_max(u)。全 batched。

    Args:
        p_tgt:    (N, K, 3) 目标位置（世界系）
        T_base:   (N, 4, 4) base 位姿（含 posture）
        T_mount:  (4, 4)    肩膀安装偏移（常数）
        table:    ReachabilityTable
        tool_dir_world: (N, K, 3) 可选，4D 表时需要

    Returns:
        rho:  (N, K)
        r:    (N, K) 实际距离
        u:    (N, K, 3) 方向向量（肩坐标系）
    """
    T_sh = T_base @ T_mount                                     # (N, 4, 4)
    p_sh = T_sh[:, :3, 3]                                       # (N, 3)
    R_sh = T_sh[:, :3, :3]                                      # (N, 3, 3)

    d = torch.einsum('nij,nkj->nki', R_sh.transpose(-1,-2),
                     p_tgt - p_sh.unsqueeze(1))                  # (N, K, 3) 肩系
    r = d.norm(dim=-1).clamp_min(1e-6)                           # (N, K)
    u = d / r.unsqueeze(-1)                                      # (N, K, 3)

    if tool_dir_world is not None:
        tool_dir_sh = torch.einsum('nij,nkj->nki', R_sh.transpose(-1,-2),
                                   tool_dir_world)
        R_max = table.query(u, tool_dir_sh)
    else:
        R_max = table.query(u)

    rho = r / R_max
    return rho, r, u


def compute_rho_profile(s_k, gamma, T_base_now, T_mount, table):
    """
    冻结 base 假设下的未来 ρ_k profile。

    Args:
        s_k:       (N, K) preview 弧长
        gamma:     TrajectoryBatch
        T_base_now:(N, 4, 4) 当前 base 位姿

    Returns:
        rho_k:     (N, K) 未来各点的 ρ
        urgency:   (N,)   视界内最大超标量
        s_to_viol: (N,)   到第一个越界点的归一化距离
    """
    p_k = gamma.p_at(s_k)                                        # (N, K, 3)
    rho_k, _, _ = compute_rho(p_k, T_base_now, T_mount, table)

    urgency = (rho_k - RHO_HI).clamp_min(0).max(dim=-1).values
    violations = rho_k > RHO_HI                                  # (N, K) bool
    # s_to_viol: 第一个 True 的索引 / K，没有则为 1.0
    first_viol = violations.float().argmax(dim=-1)
    any_viol = violations.any(dim=-1)
    s_to_viol = torch.where(any_viol, first_viol.float() / K, torch.ones_like(first_viol, dtype=torch.float))

    return rho_k, urgency, s_to_viol
```

### 测试标准

1. 建完 2D 表后可视化：切出水平/垂直截面，应该符合臂的物理形状（前方远、后方近等）
2. compute_rho 在已知构型下与解析计算吻合
3. 4D 表的切片在工具正指和反指时应有显著差异
4. GPU 性能：N=4096, K=9 时 compute_rho_profile < 0.2ms

---

## M3: 在线 Base 前馈

### 位置

新建 `modules/base_feedforward.py`

### 核心函数

```python
class OnlineBaseNom:
    """在线计算 base 前馈速度 v_ff"""

    def __init__(self, T_mount, table, rho_star=0.6,
                 T_response=0.5, filter_fc=1.5, dt=0.005):
        """
        T_mount:    (4, 4) 肩膀安装偏移
        table:      ReachabilityTable
        rho_star:   float 目标 ρ 值
        T_response: float base 响应时间常数 (s)
        filter_fc:  float v_ff 低通滤波截止频率 (Hz)
        dt:         float 控制步长 (s)
        """
        self.T_mount = T_mount
        self.table = table
        self.rho_star = rho_star
        self.T_response = T_response
        # 一阶 IIR 低通滤波器系数
        alpha = 2 * math.pi * filter_fc * dt / (2 * math.pi * filter_fc * dt + 1)
        self.filter_alpha = alpha

        # 滤波器状态
        self.v_ff_filtered = None  # (N, 3)，需要在 reset 时初始化

    def reset(self, env_ids, N, device):
        """环境 reset 时初始化滤波器状态"""
        if self.v_ff_filtered is None:
            self.v_ff_filtered = torch.zeros(N, 3, device=device)
        self.v_ff_filtered[env_ids] = 0.0

    def compute(self, s_current, gamma, p_base_current, R_base_current,
                yaw_current, preview_weights=None):
        """
        计算 v_ff 和 yaw_ff。

        Args:
            s_current:     (N,)     当前弧长
            gamma:         TrajectoryBatch
            p_base_current:(N, 3)   当前 base 位置（世界系）
            R_base_current:(N, 3, 3) 当前 base 旋转
            yaw_current:   (N,)     当前 yaw

        Returns:
            v_ff:     (N, 3) 低通滤波后的前馈速度（世界系）
            yaw_rate_ff: (N,) 前馈 yaw 速度
        """
        # 1. 采 preview 点
        L_h = 0.5  # preview horizon (m)
        Delta = L_h * torch.tensor(
            [0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.85, 1.0],
            device=s_current.device
        )
        s_k = (s_current.unsqueeze(1) + Delta.unsqueeze(0))  # (N, K)
        # clamp 到各自的 L
        s_k = s_k.clamp(max=gamma.L.unsqueeze(1))
        p_k = gamma.p_at(s_k)  # (N, K, 3)

        # 2. 每个 preview 点的理想肩膀位置
        p_sh_current = (R_base_current @ self.T_mount[:3, 3]) + p_base_current
        d_k = p_k - p_sh_current.unsqueeze(1)        # (N, K, 3)
        r_k = d_k.norm(dim=-1).clamp_min(1e-6)       # (N, K)
        u_k = d_k / r_k.unsqueeze(-1)                # (N, K, 3)
        R_max_k = self.table.query(u_k)               # (N, K)
        r_ideal_k = self.rho_star * R_max_k            # (N, K)
        p_sh_ideal_k = p_k - u_k * r_ideal_k.unsqueeze(-1)  # (N, K, 3)

        # 理想 base 位置 = 理想肩膀位置 - T_mount 平移
        mount_offset = self.T_mount[:3, 3]  # (3,)
        p_base_ideal_k = p_sh_ideal_k - (R_base_current @ mount_offset).unsqueeze(1)

        # 3. 指数衰减加权平均
        weights = torch.exp(-2.0 * Delta / L_h)       # (K,)
        weights = weights / weights.sum()
        p_base_target = (weights.unsqueeze(0).unsqueeze(-1) * p_base_ideal_k).sum(dim=1)
        # p_base_target: (N, 3)

        # 4. v_ff = 朝目标走
        v_ff_raw = (p_base_target - p_base_current) / self.T_response

        # 5. 低通滤波
        self.v_ff_filtered = (self.filter_alpha * v_ff_raw
                              + (1 - self.filter_alpha) * self.v_ff_filtered)

        # 6. yaw: 对齐 preview 窗口内目标的加权平均方向
        target_dir = (weights.unsqueeze(0).unsqueeze(-1) * d_k[..., :2]).sum(dim=1)
        target_yaw = torch.atan2(target_dir[:, 1], target_dir[:, 0])
        yaw_err = wrap_angle(target_yaw - yaw_current)
        yaw_rate_ff = yaw_err / self.T_response

        return self.v_ff_filtered, yaw_rate_ff
```

### 测试标准

1. 直线轨迹：v_ff 方向应沿轨迹方向，大小 ≈ ṡ_ref
2. 圆弧轨迹：v_ff 应指向圆弧切线方向附近
3. 突然掉头的轨迹：滤波后 v_ff 应平滑过渡
4. 静止轨迹（s_ref 常数）：v_ff ≈ 0

---

## M4: DLS-IK 求解器

### 位置

新建 `modules/dls_ik.py`

### 接口

```python
class DLSIK:
    """阻尼最小二乘逆运动学，batched GPU 实现"""

    def __init__(self, fk_fn, jac_fn, n_joints, damping=0.05,
                 max_iter=10, tol=1e-4):
        """
        fk_fn:   callable(q) -> (p, R)，正运动学
        jac_fn:  callable(q) -> J (6, n_joints)，雅可比
        damping: float DLS 阻尼系数
        """

    def solve(self, q_seed, p_target, R_target):
        """
        Args:
            q_seed:   (N, n_joints) warm start（用上一步的解）
            p_target: (N, 3)  目标位置（base frame）
            R_target: (N, 3, 3) 目标姿态（base frame）

        Returns:
            q_ik:     (N, n_joints)  IK 解
            converged:(N,) bool      是否收敛
            err:      (N,)           残余误差范数
        """
        q = q_seed.clone()
        for _ in range(self.max_iter):
            p_cur, R_cur = self.fk_fn(q)
            # 位置误差
            dp = p_target - p_cur                         # (N, 3)
            # 姿态误差：log map
            dR = R_target @ R_cur.transpose(-1, -2)
            dtheta = so3_log(dR)                          # (N, 3)
            err_vec = torch.cat([dp, dtheta], dim=-1)     # (N, 6)

            if err_vec.norm(dim=-1).max() < self.tol:
                break

            J = self.jac_fn(q)                            # (N, 6, n_joints)
            # DLS: dq = J^T (JJ^T + λ²I)^{-1} err
            JJT = J @ J.transpose(-1, -2)                 # (N, 6, 6)
            JJT_reg = JJT + self.damping**2 * torch.eye(6, device=q.device)
            dq = J.transpose(-1, -2) @ torch.linalg.solve(JJT_reg, err_vec.unsqueeze(-1))
            dq = dq.squeeze(-1)                           # (N, n_joints)

            q = q + dq

        return q, err_vec.norm(dim=-1) < self.tol * 10, err_vec.norm(dim=-1)

    @staticmethod
    def detect_branch_jump(q_ik, q_ik_prev, threshold=0.5):
        """
        检测 IK 分支跳变。

        Returns:
            jumped: (N,) bool
        """
        return (q_ik - q_ik_prev).norm(dim=-1) > threshold
```

### 注意事项

- `so3_log` 需要处理角度接近 π 的退化情况
- warm start 极其重要：每步用上一步的 q_ik 做 seed，保持分支连续
- damping 系数影响精度 vs 奇异鲁棒性的 trade-off，典型值 0.01-0.1

### 测试标准

1. 在已知构型下，FK(IK(p,R)) ≈ (p, R)
2. 在奇异点附近（臂完全伸直），输出有界（不爆炸）
3. warm start 下连续移动目标，q_ik 连续变化（不跳分支）
4. detect_branch_jump 在人为制造跳变时能检测到

---

## M5: 轨迹生成器

### 位置

新建 `modules/trajectory_generator.py`

### 核心函数

```python
class GeometryGenerator:
    """正向生成 EE 几何路径"""

    def generate(self, params):
        """
        用多频正弦叠加生成 3D 位置轨迹 + SO(3) 姿态轨迹。

        Args:
            params: dict with keys:
                f_max:        float 频率上限 (Hz)
                amplitude:    float 空间幅度 (m)
                f_rot_max:    float 旋转频率上限 (Hz)
                tangent_align_ratio: float [0,1] tangent-aligned 占比
                drift_speed:  float 漂移速度 (m/s)，最低难度也 > 0
                drift_dir:    (2,) 漂移方向 (xy)
                duration:     float 持续时间 (s)
                dt:           float 采样间隔 (s)

        Returns:
            gamma: Gamma
        """
        T = params['duration']
        dt = params['dt']
        t = torch.arange(0, T, dt)

        # 位置：多频叠加 + 漂移
        N_freqs = 8
        p = torch.zeros(len(t), 3)
        for axis in range(3):
            f_k = torch.rand(N_freqs) * params['f_max']
            A_k = (torch.rand(N_freqs) - 0.5) * 2 * params['amplitude']
            phi_k = torch.rand(N_freqs) * 2 * math.pi
            p[:, axis] = (A_k * torch.sin(2*math.pi*f_k*t.unsqueeze(1) + phi_k)).sum(dim=1)

        # 加漂移
        p[:, 0] += params['drift_dir'][0] * params['drift_speed'] * t
        p[:, 1] += params['drift_dir'][1] * params['drift_speed'] * t

        # 位置居中到工作空间中心
        p += WORKSPACE_CENTER

        # 姿态
        R = self._generate_rotation(t, params)

        # 转成标准形式
        gamma, _ = to_canonical(p, R, dt, lam=0.15)
        return gamma

    def _generate_rotation(self, t, params):
        """混合 tangent-aligned 和 independent 模式"""
        # ... 详见项目设计文档 §7.3


class TimingGenerator:
    """生成时间律 s_ref(t)"""

    def generate(self, L, f_max, v_max, T, dt):
        """
        用多频正弦叠加生成 ṡ_ref(t)。

        1. raw = Σ A_k sin(2π f_k t + φ_k)
        2. sdot = v_max * softplus(raw)    # 保证 ≥ 0
        3. sdot *= L / trapz(sdot, dt)     # 归一化到刚好走完 L
        4. s_ref = cumsum(sdot) * dt

        Returns:
            TimeLaw
        """


class TrajectoryFactory:
    """整合几何+时间律生成，并标注难度"""

    def __init__(self, base_nom_computer, table, T_mount):
        self.base_nom = base_nom_computer
        self.table = table
        self.T_mount = T_mount

    def generate_and_label(self, geom_params, timing_params):
        """
        Returns:
            gamma:      Gamma
            time_law:   TimeLaw
            difficulty: dict with keys 'v_base_max', 'a_base_max', 'v_base_mean'
        """
        gamma = self.geom_gen.generate(geom_params)
        time_law = self.timing_gen.generate(gamma.L, ...)

        # 事后标注：沿路径每隔 ds 算一次 base_nom，求导得 v/a
        s_dense = torch.arange(0, gamma.L, 0.01)
        # ... 计算 v_base_implied, a_base_implied

        difficulty = {
            'v_base_max': v_base_implied.norm(dim=-1).max().item(),
            'a_base_max': a_base_implied.norm(dim=-1).max().item(),
            'v_base_mean': v_base_implied.norm(dim=-1).mean().item(),
        }
        return gamma, time_law, difficulty

    def generate_for_level(self, level_A, level_B):
        """
        按课程等级生成。拒绝采样直到难度落在范围内。

        level_A: dict with 'v_base_range' = (lo, hi)
        level_B: dict with 'f_max', 'v_max', 'tau_range'
        """
        for _ in range(100):  # 最多尝试 100 次
            geom_params = self._sample_geom_params(level_A)
            timing_params = self._sample_timing_params(level_B)
            gamma, tl, diff = self.generate_and_label(geom_params, timing_params)
            if level_A['v_base_range'][0] <= diff['v_base_max'] <= level_A['v_base_range'][1]:
                return gamma, tl, diff
        # fallback: 返回最后一次生成的（不完全匹配但可用）
        return gamma, tl, diff
```

### 测试标准

1. 低 f_max 生成的轨迹目视应该平滑
2. 所有生成的轨迹 γ 的 L > 0
3. timing_generator 的 s_ref(T) ≈ L（走完整条路径）
4. generate_for_level 的接受率在各难度等级 > 20%

---

## M6: WTW 带臂重训

### 位置

修改 RoboDuet 现有的 leg policy 训练代码。主要改动在 env 和 reward 文件中。

### 要改的内容

**RoboDuet 的腿部 policy 不能直接用**，原因：

1. 没有经过带臂质量/惯量的训练，臂一动就 OOD
2. posture 命令范围太窄（为 locomotion 设计，不够 loco-manip 用）
3. 缺少响应一致性 reward
4. 缺少臂前馈通道
5. 缺少反向输出接口（gait phase 等）

### 改动清单

```
[修改] envs/leg_env.py
  - 加载带臂的 URDF
  - 臂关节随机运动（随机正弦轨迹，幅度和频率随机化）
  - 随机 payload（质量、CoM 偏移、惯量）
  - posture 命令范围放宽：
      height:  [-0.15, +0.10] m  （原 WTW 可能只有 [-0.05, +0.05]）
      pitch:   [-0.4, +0.3] rad
      roll:    [-0.3, +0.3] rad
      width:   [-0.05, +0.05] m
  - 固定 gait type = trot

[修改] rewards/leg_rewards.py
  - 保留原有 tracking reward
  - 添加响应一致性 reward:
      v_ref += (v_cmd - v_ref) / T * dt    # 一阶参考模型
      r_consistency = -w * ||v_actual - v_ref||
  - 不添加 base orientation / height penalty

[新增] 反向输出接口
  - gait_phase: (N, 2)  sin/cos
  - contact_state: (N, 4) 四足接触布尔
  - v_residual: (N, 3) = v_actual - v_cmd
  - base_twist_pred: (N, 6) 短期预测（可选，先不做）

[新增] 臂前馈通道（观测）
  - 臂关节角度 q_arm: (N, n_arm_joints)
  - 臂关节速度 dq_arm: (N, n_arm_joints)
  - 未来 CoM 偏移预测: (N, 3) （训练时用 ground truth）
```

### 训练完成标准

1. 速度跟踪误差（所有方向）< 0.1 m/s RMS
2. 响应一致性：不同 payload/臂运动下，v_actual 和参考模型 v_ref 的偏差 < 0.05 m/s RMS
3. 不摔（全 posture 范围 + 全 payload 范围内稳定）
4. **冻结，后续不再更新**

---

## M7: 上层环境封装

### 位置

新建 `envs/tracking_env.py`，继承 RoboDuet 的 base env 或 Isaac Gym 的 VecEnv。

### 核心结构

```python
class TrackingEnv:
    """上层 policy 的训练环境"""

    def __init__(self, cfg):
        # 加载物理仿真（Isaac Gym）
        # 加载冻结的腿部 policy（M6 的 checkpoint）
        # 初始化所有模块
        self.trajectory_batch = TrajectoryBatch(N, max_points, device)
        self.reach_table = ReachabilityTable.load(cfg.reach_table_path)
        self.base_nom = OnlineBaseNom(T_mount, self.reach_table, ...)
        self.ik = DLSIK(fk_fn, jac_fn, ...)
        self.traj_factory = TrajectoryFactory(...)
        self.curriculum = CurriculumManager(...)

        # 状态
        self.s = torch.zeros(N, device=device)       # 当前弧长
        self.q_ik_prev = torch.zeros(N, n_arm, device=device)
        self.ema_dq = torch.zeros(N, n_arm, device=device)
        self.rho_prev = torch.zeros(N, device=device)

    def reset(self, env_ids):
        """
        1. 从课程管理器获取难度等级
        2. 调用 TrajectoryFactory 生成轨迹
        3. 写入 trajectory_batch
        4. 重置状态变量 (s, q_ik_prev, ema_dq, ...)
        5. 采样 τ（time tube 半径）
        6. 重置 base_nom 的滤波器
        """

    def step(self, actions):
        """
        一个上层 policy step 的完整流程。

        actions: dict with keys
            'dq_arm':   (N, n_arm)    臂关节残差
            'dv':       (N, 3)        v_ff 残差
            'posture':  (N, 4)        h/pitch/roll/width 命令
            'gait':     (N, 2)        步频 / swing height
        """
        # 1. 限幅和限速
        dq_arm = torch.tanh(actions['dq_arm']) * DQ_MAX          # ~5° = 0.087 rad
        dv = torch.tanh(actions['dv']) * DV_MAX                  # ~0.3 m/s
        posture = rate_limit(actions['posture'], self.posture_prev, POSTURE_RATE_MAX)
        gait_params = rate_limit(actions['gait'], self.gait_prev, GAIT_RATE_MAX)

        # 2. 计算 v_ff
        v_ff, yaw_rate_ff = self.base_nom.compute(
            self.s, self.trajectory_batch,
            self.base_pos, self.base_rot, self.yaw
        )

        # 3. 组装 v_cmd
        v_cmd = v_ff + dv

        # 4. 目标变换到 base frame → IK
        s_ref_now = self.trajectory_batch.s_ref(self.sim_time)
        p_tgt_world = self.trajectory_batch.gamma_p_at(self.s)
        R_tgt_world = self.trajectory_batch.gamma_R_at(self.s)
        p_tgt_base, R_tgt_base = world_to_base(p_tgt_world, R_tgt_world,
                                                self.base_pos, self.base_rot)
        q_ik, converged, ik_err = self.ik.solve(self.q_ik_prev, p_tgt_base, R_tgt_base)

        # 5. IK 分支跳变检测
        jumped = DLSIK.detect_branch_jump(q_ik, self.q_ik_prev)

        # 6. 臂目标 = IK + 残差
        q_arm_target = q_ik + dq_arm

        # 7. 组装腿部 policy 的输入并推理
        leg_obs = self._build_leg_obs(v_cmd, posture, gait_params, q_arm_target)
        with torch.no_grad():
            leg_actions = self.leg_policy(leg_obs)

        # 8. 发送到仿真
        #    臂: q_arm_target → PD 控制器（高频，可在物理步内插值）
        #    腿: leg_actions → PD 控制器
        self._apply_actions(q_arm_target, leg_actions)

        # 9. 物理仿真步进（可能多个 substep）
        self.sim.step()

        # 10. 读取新状态
        self._update_state()

        # 11. 更新 s（前向窗口投影）
        self.s, self.d_lat = update_s(self.s, self.ee_pos, self.ee_rot,
                                       self.trajectory_batch)

        # 12. 更新 EMA
        dq_actual = self.arm_joint_vel * self.dt
        self.ema_dq = 0.95 * self.ema_dq + 0.05 * dq_actual

        # 13. 计算观测、reward、done
        obs = self._compute_obs()
        rewards = self._compute_rewards(jumped)
        dones = self._compute_dones(jumped)

        # 14. 更新状态变量
        self.q_ik_prev = q_ik
        self.rho_prev = self.rho_current
        self.posture_prev = posture
        self.gait_prev = gait_params

        return obs, rewards, dones, infos
```

### 频率处理

上层 policy 运行在 ~50Hz（和腿相同频率），但臂的 PD 控制器在 200Hz 的物理仿真循环内插值执行。具体来说：每个 policy step 对应 4 个物理 substep，每个 substep 内臂目标做线性插值。

如果需要更高的臂控制频率（比如 policy 也运行在 200Hz），在 step 函数内加一层循环。但 50Hz + IK 插值在大多数情况下够用，先从这里开始。

---

## M8: 观测 / 动作 / Reward

### 位置

在 `envs/tracking_env.py` 内部，或拆分到 `envs/tracking_obs.py` 和 `envs/tracking_reward.py`。

### 观测构建

```python
def _compute_obs(self):
    """参考项目设计文档 §8.1"""

    # --- 当前误差（base frame）---
    pos_err = self.ee_pos_base - self.target_pos_base     # (N, 3)
    rot_err = so3_log(self.target_R_base.T @ self.ee_R_base)  # (N, 3)
    twist_err_lin = ...                                    # (N, 3)
    twist_err_ang = ...                                    # (N, 3)

    # --- 进度 / 时序 ---
    s_normalized = self.s / self.trajectory_batch.L        # (N,)
    timing_err = self.s - self.trajectory_batch.s_ref(self.sim_time)  # (N,)
    sdot = ...                                             # (N,) 当前弧长速度
    sdot_ref = self.trajectory_batch.sdot_ref(self.sim_time)
    tau_obs = self.tau / (self.tau + 0.1)                  # (N,) τ 编码

    # --- Preview（heading frame）---
    s_k, p_k, R_k, sdot_k = self.trajectory_batch.sample_preview(self.s, L_h=0.5, K=9)
    p_k_heading = world_to_heading(p_k, self.heading_frame)     # (N, K, 3)
    R_k_6d = R_k[..., :2].reshape(N, K, 6)                     # (N, K, 6)
    tangent_k = self.trajectory_batch.tangent_at(s_k)           # (N, K, 3)
    tangent_k_heading = rotate_to_heading(tangent_k)
    rho_k, urgency, s_to_viol = compute_rho_profile(
        s_k, self.trajectory_batch, self.T_base_current, T_MOUNT, self.reach_table
    )

    # preview 拼接: (N, K, 14)
    preview = torch.cat([p_k_heading, R_k_6d, tangent_k_heading, sdot_k.unsqueeze(-1),
                         rho_k.unsqueeze(-1)], dim=-1)
    preview_flat = preview.reshape(N, -1)                       # (N, K*14)

    # --- 臂 ---
    arm_q = self.arm_joint_pos                                  # (N, n_arm)
    arm_dq = self.arm_joint_vel                                 # (N, n_arm)

    # --- base（heading frame）---
    base_rp = torch.stack([self.base_roll, self.base_pitch], dim=-1)
    base_h = self.base_height.unsqueeze(-1)
    base_twist_heading = ...                                    # (N, 6)
    v_residual = self.v_actual - self.v_cmd_prev                # (N, 3)

    # --- 步态 ---
    gait_phase = torch.stack([torch.sin(self.phase), torch.cos(self.phase)], dim=-1)
    contact = self.foot_contact                                  # (N, 4)

    # --- 构型 ---
    manip = self._compute_manipulability()                      # (N,)
    jl_dist = self._joint_limit_distance()                      # (N, n_arm)
    rho0 = rho_k[:, 0]                                         # (N,) 当前点

    # --- 协调诊断 ---
    ema_dq = self.ema_dq                                        # (N, n_arm)

    # --- 上一步动作 ---
    prev_action = self.prev_action                              # (N, dim_a)

    # 拼接
    obs = torch.cat([
        pos_err, rot_err, twist_err_lin, twist_err_ang,         # 12
        s_normalized.unsqueeze(-1), timing_err.unsqueeze(-1),
        sdot.unsqueeze(-1), sdot_ref.unsqueeze(-1),
        tau_obs.unsqueeze(-1),                                  # 5
        preview_flat,                                           # K*14
        urgency.unsqueeze(-1), s_to_viol.unsqueeze(-1),        # 2
        arm_q, arm_dq,                                          # 2*n_arm
        base_rp, base_h, base_twist_heading, v_residual,        # ~13
        gait_phase, contact,                                    # 6
        manip.unsqueeze(-1), jl_dist, rho0.unsqueeze(-1),      # ~n_arm+2
        ema_dq,                                                 # n_arm
        prev_action,                                            # dim_a
    ], dim=-1)

    return obs
```

### Reward 构建

```python
def _compute_rewards(self, jumped):
    """三组 reward，参考项目设计文档 §8.3 和 §9"""
    rewards = {}

    # === Group T: Tracking ===
    rewards['progress'] = self.w_p * self.sdot
    rewards['lateral_err'] = -self.w_e * self.d_lat
    rewards['timing'] = -self.w_t * deadzone(self.timing_err, self.tau)
    rewards['twist_err'] = -self.w_v * self.twist_err.norm(dim=-1)

    # === Group R: Reachability / 构型 ===
    rho0 = self.rho_current
    rewards['rho_barrier'] = -(
        F.softplus((rho0 - RHO_HI) / BETA) +
        F.softplus((RHO_LO - rho0) / BETA)
    ) * self.w_rho
    rewards['manipulability'] = self.w_m * self.manipulability
    rewards['joint_limit'] = -self.w_l * joint_limit_barrier(self.arm_q, Q_LO, Q_HI)
    # 协调性 reward
    rewards['ema_effort'] = -self.w_ema * self.ema_dq.norm(dim=-1)
    rewards['rho_rate'] = -self.w_rr * (rho0 - self.rho_prev).abs() / self.dt

    # === Group S: Smoothness ===
    rewards['action_rate'] = -self.w_ar * (self.action - self.prev_action).norm(dim=-1)
    rewards['dq_magnitude'] = -self.w_dq * self.dq_arm_action.norm(dim=-1)
    rewards['arm_accel'] = -self.w_acc * self.arm_joint_acc.norm(dim=-1)
    rewards['posture_rate'] = -self.w_pr * (self.posture - self.posture_prev).norm(dim=-1)
    rewards['dv_magnitude'] = -self.w_dv * self.dv_action.norm(dim=-1)

    # 分组求和（multi-critic 需要分组 reward）
    r_T = sum(rewards[k] for k in ['progress', 'lateral_err', 'timing', 'twist_err'])
    r_R = sum(rewards[k] for k in ['rho_barrier', 'manipulability', 'joint_limit',
                                    'ema_effort', 'rho_rate'])
    r_S = sum(rewards[k] for k in ['action_rate', 'dq_magnitude', 'arm_accel',
                                    'posture_rate', 'dv_magnitude'])

    return {'total': r_T + r_R + r_S, 'group_T': r_T, 'group_R': r_R, 'group_S': r_S,
            **rewards}  # 保留细项用于 logging
```

### 辅助函数

```python
def deadzone(err, tau):
    """死区惩罚：|err| < tau 时为 0，超出部分线性或二次增长"""
    return F.relu(err.abs() - tau)

def joint_limit_barrier(q, q_lo, q_hi, margin=0.1):
    """关节限位软 barrier"""
    return (F.softplus((q_lo + margin - q) / 0.02) +
            F.softplus((q - q_hi + margin) / 0.02)).sum(dim=-1)

def rate_limit(cmd, cmd_prev, max_rate):
    """限速"""
    delta = (cmd - cmd_prev).clamp(-max_rate, max_rate)
    return cmd_prev + delta
```

---

## M9: Multi-Critic PPO

### 位置

新建 `algorithms/multi_critic_ppo.py`，或修改 RoboDuet 现有的 PPO。

### 核心改动

RoboDuet 的 PPO 用一个 value head。我们需要三个 value head，各自估计各组 reward 的 return。

```python
class MultiCriticActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden=[512, 256, 128]):
        super().__init__()
        # 共享特征提取
        self.shared = MLP(obs_dim, hidden[0], hidden[0])

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.Linear(hidden[1], act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # 可选：per-channel 初始 std（§M8 动作空间中 base 通道 std 设大）
        # self.log_std.data[dv_indices] = -0.2   # base 通道 std ≈ 0.82
        # self.log_std.data[dq_indices] = -1.0   # arm 通道 std ≈ 0.37

        # 三个 Critic
        self.critic_T = MLP(obs_dim, hidden[1], 1)
        self.critic_R = MLP(obs_dim, hidden[1], 1)
        self.critic_S = MLP(obs_dim, hidden[1], 1)

    def forward(self, obs):
        features = self.shared(obs)
        mean = self.actor(features)
        std = self.log_std.exp()
        return mean, std

    def evaluate(self, obs):
        v_T = self.critic_T(obs).squeeze(-1)
        v_R = self.critic_R(obs).squeeze(-1)
        v_S = self.critic_S(obs).squeeze(-1)
        return v_T, v_R, v_S


class MultiCriticPPO:

    def __init__(self, model, cfg):
        self.model = model
        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.clip_ratio = cfg.clip_ratio
        # 三组 reward 的权重（policy gradient 层面）
        self.group_weights = {'T': cfg.w_T, 'R': cfg.w_R, 'S': cfg.w_S}

    def compute_advantages(self, rollout_buffer):
        """
        对每组 reward 分别计算 GAE advantage，各自 normalize，
        然后加权求和得到 total advantage。
        """
        adv_T = gae(rollout_buffer['group_T'], rollout_buffer['values_T'],
                     rollout_buffer['dones'], self.gamma, self.lam)
        adv_R = gae(rollout_buffer['group_R'], rollout_buffer['values_R'],
                     rollout_buffer['dones'], self.gamma, self.lam)
        adv_S = gae(rollout_buffer['group_S'], rollout_buffer['values_S'],
                     rollout_buffer['dones'], self.gamma, self.lam)

        # 各自 normalize
        adv_T = (adv_T - adv_T.mean()) / (adv_T.std() + 1e-8)
        adv_R = (adv_R - adv_R.mean()) / (adv_R.std() + 1e-8)
        adv_S = (adv_S - adv_S.mean()) / (adv_S.std() + 1e-8)

        # 加权求和
        advantages = (self.group_weights['T'] * adv_T +
                      self.group_weights['R'] * adv_R +
                      self.group_weights['S'] * adv_S)

        return advantages, {'T': adv_T, 'R': adv_R, 'S': adv_S}

    def update(self, rollout_buffer):
        """标准 PPO update，但用 multi-critic 的 advantage"""
        advantages, adv_groups = self.compute_advantages(rollout_buffer)
        # ... 标准 PPO 的 policy loss + 三个 value loss
```

---

## M10: 课程管理器

### 位置

新建 `modules/curriculum.py`

### 核心类

```python
class CurriculumManager:
    """二维网格课程：(A: base 运动需求, B: 时序带宽)"""

    def __init__(self, n_levels_A=8, n_levels_B=6,
                 success_threshold=0.7, fail_threshold=0.3):
        # 每个 cell 记录成功率
        self.success_rates = torch.zeros(n_levels_A, n_levels_B)
        self.episode_counts = torch.zeros(n_levels_A, n_levels_B)

        # 等级定义
        self.levels_A = [
            {'v_base_range': (0.0, 0.1), 'drift_speed_range': (0.02, 0.05)},
            {'v_base_range': (0.05, 0.2), 'drift_speed_range': (0.05, 0.1)},
            {'v_base_range': (0.1, 0.4), 'drift_speed_range': (0.1, 0.2)},
            # ... 逐步提高
        ]
        self.levels_B = [
            {'f_max': 0.5, 'v_max': 0.1, 'tau_range': (0.5, 2.0)},
            {'f_max': 1.0, 'v_max': 0.2, 'tau_range': (0.2, 1.0)},
            {'f_max': 2.0, 'v_max': 0.4, 'tau_range': (0.05, 0.5)},
            # ... 逐步提高
        ]

        # 每个环境当前在哪个 cell
        # 初始化略

    def get_level(self, env_id):
        """返回该环境当前的 (level_A, level_B)"""
        i, j = self.current_cell[env_id]
        return self.levels_A[i], self.levels_B[j]

    def report_result(self, env_id, success):
        """
        episode 结束时报告成功/失败。
        成功条件：d_lat < threshold && timing_err < threshold && 没有 IK 跳变 && 走完 > 80% 路径
        """
        i, j = self.current_cell[env_id]
        self.episode_counts[i, j] += 1
        self.success_rates[i, j] = (
            0.95 * self.success_rates[i, j] + 0.05 * float(success)
        )

        # 网格推进
        if self.success_rates[i, j] > self.success_threshold:
            self._promote(env_id)
        elif self.success_rates[i, j] < self.fail_threshold:
            self._demote(env_id)

    def sample_with_history_mixing(self, env_id):
        """
        60% 概率从当前前沿采样，40% 从历史已解锁等级均匀采样。
        返回 (level_A, level_B, tau)
        """
        if random.random() < 0.6:
            level_A, level_B = self.get_level(env_id)
        else:
            # 从所有已解锁 cell 中均匀采
            unlocked = self._get_unlocked_cells(env_id)
            i, j = random.choice(unlocked)
            level_A, level_B = self.levels_A[i], self.levels_B[j]

        # 采 τ
        tau_lo, tau_hi = level_B['tau_range']
        # 混合分布
        r = random.random()
        if r < 0.2:
            tau = 0.0
        elif r < 0.4:
            tau = tau_hi  # 大 τ ≈ 纯路径跟随
        else:
            tau = math.exp(random.uniform(math.log(max(tau_lo, 0.01)), math.log(tau_hi)))

        return level_A, level_B, tau
```

---

## M11: 安全滤波器（可选）

### 位置

新建 `modules/safety_filter.py`

### 核心类

```python
class ReachabilitySafetyFilter:
    """基于 ρ 梯度的 CBF 风格安全滤波器"""

    def __init__(self, rho_hi=0.85, rho_lo=0.35, kappa=3.0,
                 hysteresis_on=0.02, hysteresis_off=0.05,
                 smoothing_alpha=0.1):
        self.rho_hi = rho_hi
        self.rho_lo = rho_lo
        self.kappa = kappa
        self.active = None    # (N,) bool

    def reset(self, env_ids, N, device):
        if self.active is None:
            self.active = torch.zeros(N, dtype=torch.bool, device=device)
        self.active[env_ids] = False

    def filter(self, a, rho0, rho_dot_drift, grad_rho_a):
        """
        Args:
            a:              (N, dim_a) policy 输出的名义动作
            rho0:           (N,)       当前 ρ
            rho_dot_drift:  (N,)       ρ 因目标运动的变化率（不可控部分）
            grad_rho_a:     (N, dim_a) ρ 对动作 a 的梯度

        Returns:
            a_filtered:     (N, dim_a) 滤波后的动作
            correction:     (N, dim_a) 修正量（用于喂回观测）
        """
        rho_dot = rho_dot_drift + (grad_rho_a * a).sum(dim=-1)  # (N,)

        # --- 上界 ---
        margin_hi = self.kappa * (self.rho_hi - rho0)
        viol_hi = (rho_dot - margin_hi).clamp_min(0)

        g = grad_rho_a
        g_norm_sq = (g * g).sum(dim=-1).clamp_min(1e-8)
        correction_hi = (viol_hi / g_norm_sq).unsqueeze(-1) * g

        # --- 下界（对称）---
        margin_lo = -self.kappa * (rho0 - self.rho_lo)
        rho_dot_after_hi = rho_dot - (grad_rho_a * correction_hi).sum(dim=-1)
        viol_lo = (margin_lo - rho_dot_after_hi).clamp_min(0)
        correction_lo = -(viol_lo / g_norm_sq).unsqueeze(-1) * g

        correction = correction_hi + correction_lo

        # --- 滞回 ---
        should_activate = (rho0 > self.rho_hi - self.hysteresis_on) | \
                          (rho0 < self.rho_lo + self.hysteresis_on)
        should_deactivate = (rho0 < self.rho_hi - self.hysteresis_off) & \
                            (rho0 > self.rho_lo + self.hysteresis_off)
        self.active = self.active | should_activate
        self.active = self.active & ~should_deactivate

        # 只在 active 时应用修正
        correction = correction * self.active.unsqueeze(-1).float()

        # --- 低通平滑 ---
        # (需要维护 self.correction_prev 状态)

        a_filtered = a - correction
        return a_filtered, correction
```

### 梯度计算

ρ 对动作 a 的梯度需要通过以下链条计算：

```
a → ξ̇ (base 速度 + posture 速率) → T_shoulder 变化 → d 变化 → ρ 变化
```

最简单的实现：用 torch.autograd

```python
def compute_rho_grad(rho0_fn, xi, a):
    """
    rho0_fn: 给定 xi 返回 rho0 的可微函数
    xi:      当前 base 状态 (N, 6)
    a:       动作 (N, dim_a)，其中 base 相关分量映射到 xi_dot

    注意：只需要对 a 中的 base 速度和 posture 分量求梯度，
    arm 分量的梯度为 0（ρ 不依赖臂关节角）。
    """
    xi_var = xi.clone().requires_grad_(True)
    rho0 = rho0_fn(xi_var)
    grad_xi = torch.autograd.grad(rho0.sum(), xi_var, create_graph=False)[0]
    # grad_rho_a = grad_xi @ (dxi/da)，后者是线性映射
    # ...
```

---

## M12: 评测脚本

### 位置

新建 `evaluation/eval_suite.py`

### 要实现的评测

```python
class EvalSuite:
    """完整评测套件"""

    def bandwidth_sweep(self, policy, env, freqs, amplitude=0.05):
        """
        闭环带宽测量。

        对每个频率 f：
        1. 生成正弦轨迹 p(t) = A·sin(2πft)（单轴）
        2. 运行 policy
        3. 测量 EE 响应的幅度和相位延迟
        4. 画幅频曲线（Bode 图）

        -3dB 截止频率 = 闭环带宽
        """

    def error_spectrum(self, policy, env, trajectories):
        """
        对每条轨迹：
        1. 运行 policy，记录 EE 误差时间序列
        2. FFT → 功率谱密度
        3. 诊断：2-4Hz 有峰？低频漂移？宽带噪声？
        """

    def tau_pareto(self, policy, env, trajectories, tau_values):
        """
        固定轨迹集，扫 τ 从 0 到 τ_max。
        对每个 τ：运行 policy，记录 (mean_timing_err, mean_lateral_err)。
        画 Pareto 曲线。
        """

    def workspace_volume(self, policy, env, eps=0.02):
        """
        有效工作空间：在 base 周围密集撒目标点，
        policy 能够在 EE 误差 < eps 下到达的点的集合。
        """

    def base_utilization(self, policy, env, trajectories):
        """
        base 利用率 = ||v_base_actual|| / ||v_ff||
        长期 ≪ 1 = policy 退化了
        """

    def rho_distribution(self, policy, env, trajectories):
        """
        统计 ρ 的分布。P(ρ > ρ_hi) 应该很低。
        另外统计"舒适带内失败率"：P(跟踪失败 | ρ ∈ [ρ_lo, ρ_hi])。
        """

    def ik_branch_jump_rate(self, policy, env, trajectories):
        """
        统计 IK 分支跳变率。应趋近 0。
        """

    def full_report(self, policy, env, held_out_synthetic, held_out_mocap):
        """跑完所有指标，输出表格 + 图"""
        results = {}
        results['bandwidth'] = self.bandwidth_sweep(policy, env, ...)
        results['spectrum'] = self.error_spectrum(policy, env, held_out_synthetic)
        results['tau_pareto'] = self.tau_pareto(policy, env, held_out_synthetic, ...)
        results['base_util'] = self.base_utilization(policy, env, held_out_synthetic)
        results['rho_dist'] = self.rho_distribution(policy, env, held_out_synthetic)
        results['ik_jumps'] = self.ik_branch_jump_rate(policy, env, held_out_synthetic)
        # 动捕集
        results['mocap_tracking'] = self.error_spectrum(policy, env, held_out_mocap)
        return results
```

---

## 附录：超参数参考表

以下为建议的初始值，需要实验调整。

### 轨迹

| 参数    | 值         | 说明                          |
| ------- | ---------- | ----------------------------- |
| λ      | 0.15 m/rad | SE(3) 弧长的旋转-平移等价系数 |
| ds_grid | 0.005 m    | 几何路径采样间隔              |
| L_h     | 0.5 m      | preview horizon               |
| K       | 9          | preview 点数                  |

### 可达性

| 参数       | 值   | 说明                   |
| ---------- | ---- | ---------------------- |
| ρ_lo      | 0.35 | 舒适带下界             |
| ρ_hi      | 0.85 | 舒适带上界             |
| ρ*        | 0.6  | base_nom 的目标 ρ     |
| beta       | 0.05 | barrier 过渡陡峭度     |
| σ_rot_min | 0.05 | 建表时旋转奇异过滤阈值 |

### 在线 base_nom

| 参数       | 值     | 说明              |
| ---------- | ------ | ----------------- |
| T_response | 0.5 s  | base 响应时间常数 |
| filter_fc  | 1.5 Hz | v_ff 低通截止频率 |

### 动作限幅

| 参数             | 值                    | 说明                                     |
| ---------------- | --------------------- | ---------------------------------------- |
| DQ_MAX           | 0.087 rad (~5°)      | 臂残差限幅                               |
| DV_MAX           | 0.3 m/s               | v_ff 残差限幅                            |
| POSTURE_RATE_MAX | [0.1, 0.2, 0.2, 0.05] | h/pitch/roll/width 变化率上限 (per step) |

### EMA

| 参数      | 值   | 说明                                |
| --------- | ---- | ----------------------------------- |
| alpha_ema | 0.95 | EMA 系数，对应 ~0.4s 时间常数 @50Hz |

### Reward 权重（初始值，需调）

| 组 | 项             | 符号  | 初始值 |
| -- | -------------- | ----- | ------ |
| T  | progress       | w_p   | 1.0    |
| T  | lateral_err    | w_e   | 5.0    |
| T  | timing         | w_t   | 2.0    |
| T  | twist_err      | w_v   | 1.0    |
| R  | rho_barrier    | w_ρ  | 2.0    |
| R  | manipulability | w_m   | 0.5    |
| R  | joint_limit    | w_l   | 1.0    |
| R  | ema_effort     | w_ema | 1.0    |
| R  | rho_rate       | w_rr  | 0.5    |
| S  | action_rate    | w_ar  | 0.1    |
| S  | dq_magnitude   | w_dq  | 0.2    |
| S  | arm_accel      | w_acc | 0.05   |
| S  | posture_rate   | w_pr  | 0.5    |
| S  | dv_magnitude   | w_dv  | 0.3    |

### Multi-Critic 组权重

| 组 | 权重 |
| -- | ---- |
| T  | 1.0  |
| R  | 0.5  |
| S  | 0.3  |

### PPO

| 参数            | 值       |
| --------------- | -------- |
| γ              | 0.99     |
| λ_GAE          | 0.95     |
| clip_ratio      | 0.2      |
| learning_rate   | 3e-4     |
| n_envs          | 4096     |
| horizon         | 48 steps |
| mini_batch_size | 4096     |
| n_epochs        | 5        |

---

## 附录：Logging 清单

训练时必须记录以下量，用于诊断和论文图表：

```
# 每步
- reward 各项细分
- ρ0, ρ_k 的均值/最大值
- ||ema_dq||
- ||Δv||, ||dq_arm||
- d_lat, timing_err
- base 速度 (v_cmd, v_actual, v_ff)
- IK 残余误差

# 每 episode
- 成功/失败
- 总进度 s_final / L
- 课程等级 (i, j)
- IK 分支跳变次数
- 最大 ρ
- base 利用率

# 每 N 个 episode
- 课程网格的成功率热力图
- ρ 分布直方图
- base 利用率分布
```
