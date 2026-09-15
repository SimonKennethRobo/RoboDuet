"""Audit saved I_Q trials and render the identification handoff artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from identify_iq_mujoco import ROOT, CHANNELS, sha, write_json


def main(root):
    prediction = json.loads((root/"independent_prediction_validation.json").read_text())
    protocol = json.loads((root/"closed_loop_v2_protocol.json").read_text())
    rows, traces = [], {}
    for trial in protocol["trials"]:
        name = f"{trial['scenario']}_{trial['controller']}_s{trial['seed']}"
        folder = root/"closed_loop_v2"/name
        info = json.loads((folder/"receipt.json").read_text())
        if info["steps"]:
            if sha(folder/"trace.npz") != info["trace_sha256"]:
                raise ValueError(f"trace changed: {name}")
            traces[name] = dict(np.load(folder/"trace.npz"))
        rows.append(info)
    groups = []
    for scenario in ["hold", "long_curve", "walking_curve"]:
        for controller in ["ideal", "first_order", "gait"]:
            selected = [r for r in rows if r["scenario"]==scenario and r["controller"]==controller]
            valid = [r for r in selected if r["steps"]]
            groups.append(dict(scenario=scenario, controller=controller, requested=len(selected),
                completed=sum(r["success"] for r in selected),
                failure_ids=[r["seed"] for r in selected if not r["success"]],
                position_rmse_m=float(np.mean([r["position_rmse_m"] for r in valid])),
                orientation_rmse_rad=float(np.mean([r["orientation_rmse_rad"] for r in valid])),
                command_clipped_steps=sum(r["command_clipped_steps"] for r in selected)))
    write_json(root/"closed_loop_summary.json", dict(groups=groups, trials=rows,
        definition="mean per-trial RMSE; completion is execution without failure, not a tracking-success threshold",
        protocol_sha256=sha(root/"closed_loop_v2_protocol.json"),
        scope="3 trajectories x 2 translated starts x 3 models, 12 s each; not independent terrain/payload seeds"))
    f0 = json.loads((root/"models_v2/F0.json").read_text())
    f1 = json.loads((root/"models_v2/F1_gait.json").read_text())
    lines = ["# I_Q 系统辨识与原生 OCS2 集成（2026-09-15）", "",
        "已完成 MuJoCo 数据采集、六通道一阶模型与步态残差拟合、原生 C++ OCS2 接入和闭环对照。默认入口使用一阶模型；带残差模型可显式选择。", "",
        "## 运行", "", "```bash", "cd /home/simon/Projects/WBC/RoboDuet",
        "bash tmp/run_iq_mpc.sh", "# 带步态残差", "bash tmp/run_iq_mpc.sh --controller gait",
        "# 无窗口", "IQ_VIEWER=0 bash tmp/run_iq_mpc.sh", "```", "",
        "默认运行 12 秒、前移 1.2 m 的平滑 SE(3) 曲线。每次写入新的 interactive 子目录。"
        "该入口调用原生 C++ OCS2 SQP/MRT，MuJoCo 侧使用 Python rl_sar 观测/推理镜像与导出的 TorchScript；不是完整 ROS/FSM 或硬件验收。", "",
        "## 与当前论文的对应", "",
        "依据 `wbc_rl_mpc/overleaf/3method.tex` 的 16 状态、12 输入设计：",
        "- 状态：臂安装点世界坐标 xyz、ZYX yaw/pitch/roll、heading vx/vy、body wz、实测步态相位、6 个臂关节位置。",
        "- 输入：vx/vy/wz 指令、高度偏移/pitch/roll 指令、6 个臂关节速度。",
        "- 一阶响应：`y_dot = (gain*u + bias - y)/tau`；高度 bias 包含安装点零指令高度。",
        "- 姿态残差：`delta=(a0+a1*sqrt(vx^2+vy^2+1e-6))*sin(n*phi+psi)`。真实姿态状态包含残差；动力学对 `y-delta` 应用滞后并加回 `delta_dot`，包括速度变化引起的幅值导数。",
        "- 相位由实际策略时钟测量；2.75 Hz，速度指令前三维范数小于 0.1 时停止。桥接端展开相位，避免 2π 回绕破坏 MPC 热启动。",
        "- 臂模型沿用论文的关节速度积分器；真实 MuJoCo 臂执行 PD、限矩和目标速度限制。没有额外拟合臂动力学。", "",
        "## 采集与独立验证", "",
        "原始采集 72 条 × 20 秒，36 条拟合、18 条开发选谐波、18 条原始测试；另补采 18 条 × 20 秒独立验证，总计 30 分钟有效仿真（每条另有 3 秒预热）。90 条均完成，无数值警告。",
        "原始开发/测试中的 12 条稳态轨迹与训练激励相同，只改变平面初始位置，不能充当独立泛化证据。下表仅使用补采的独立验证：新 PRBS、新稳态速度、新起始步态相位、新机械臂振荡相位/频率；模型参数及谐波在采集前已冻结。",
        "预测只使用窗口起点测量和未来指令，递推速度和相位，不偷用未来真实状态。验证分固定臂/运动臂各 9 条，完整分组数据见 `independent_prediction_validation.json`。",
        "MuJoCo 3.3.6，物理步长 2.5 ms，控制调度 5 ms，策略 20 ms；每个物理子步重算显式 PD 力矩。该执行器语义属于本次辨识 plant，不可直接套用于已改成 MuJoCo position drive 的其他 benchmark 适配器。", "",
        "## 一阶参数 F0", "", "| 通道 | gain | tau (s) | bias |", "|---|---:|---:|---:|"]
    for name in CHANNELS:
        c=f0[name]; lines.append(f"| {name} | {c['gain']:.6f} | {c['tau_s']:.6f} | {c['bias']:.6f} |")
    lines += ["", "高度 tau 命中 3 秒拟合上界。站立/行走间存在慢漂移及增益变化，因此这个通道是受限的整体近似，不能把 3 秒解释为已精确辨识的物理时间常数。", "",
        "## 步态残差参数", "", "| 通道 | n | a0 | a1 | psi (rad) |", "|---|---:|---:|---:|---:|"]
    for name in CHANNELS[3:]:
        c=f1[name]["residual"]; lines.append(f"| {name} | {c['harmonic']} | {c['amplitude']:.7f} | {c['speed_amplitude']:.7f} | {c['phase_offset']:.6f} |")
    lines += ["", "F1_gait 的姿态 nominal 参数与残差联合拟合，其完整 gain/tau/bias 见 `models_v2/F1_gait.json`；速度三通道与 F0 相同。负幅值与相位可互换，不代表负振荡能量。", "",
        "## 独立预测 RMSE", "", "| 预测时域 | 通道 | F0 | F1_gait | 降低 |", "|---|---|---:|---:|---:|"]
    for horizon in ["0.1", "0.3", "0.6", "1.0"]:
        for c,unit in [("height","mm"),("pitch","mrad"),("roll","mrad")]:
            a=prediction["test"]["F0"][horizon][c]*1000; b=prediction["test"]["F1_gait"][horizon][c]*1000
            lines.append(f"| {horizon} s | {c} ({unit}) | {a:.3f} | {b:.3f} | {100*(1-b/a):.1f}% |")
    lines += ["", "![prediction](prediction.png)", "", "## 原生 MPC 闭环", "",
        "冻结的 3 场景 × 2 个平面平移初始位置 × 3 个模型，每次 12 秒。两次重放没有改变地形/载荷，不作随机鲁棒性统计；所有条目及失败均保留。下表为每次 RMSE 的算术平均。", "",
        "| 场景 | 模型 | 执行完成 | EE位置 RMSE (mm) | EE姿态 RMSE (rad) | 外部命令裁剪步数 |",
        "|---|---|---:|---:|---:|---:|"]
    for r in groups:
        lines.append(f"| {r['scenario']} | {r['controller']} | {r['completed']}/{r['requested']} | {1000*r['position_rmse_m']:.3f} | {r['orientation_rmse_rad']:.5f} | {r['command_clipped_steps']} |")
    lines += ["", "hold 为定点保持；long_curve 前移 0.55 m；walking_curve 前移 1.2 m，并叠加侧向/高度与俯仰小幅变化。",
        "各模型使用相同 EE 权重、臂姿态代价、自碰撞约束和外部 RL 指令裁剪。一阶/残差模型在优化内部直接约束指令；理想积分模型通过原有桥接转换后可能裁剪，因此这组数值是配置后端到端对照，不是仅改变一个动力学项的严格消融。没有宣称硬实时、硬件或 SOTA。",
        "原始任务文件的安装点高度上限 0.35 m 低于本次站立高度；三个对照副本统一改为安装点 z∈[0.27,0.47] m、pitch±0.30、roll±0.20。原任务文件未被本次修改。",
        "默认选 F0：移动轨迹位置误差较小；F1_gait 在 walking_curve 的姿态误差较小，离线预测改善没有保证每项闭环指标都改善。", "",
        "![closed loop](closed_loop.png)", "", "## 接口与使用范围", "",
        "- `mpc/task_I_Q_{ideal,first_order,gait}.info` 是已运行的任务副本；同时安装到 MPC 仓库 `go2_x5_ocs2/config/I_Q/`。",
        "- benchmark request 的 flags bit 0 增加末尾 float64 实测相位；旧 StateMsg/CmdMsg 字节布局保持。带残差模型缺少相位时拒绝运行，不能直接用旧 ROS/rl_sar v1 消息假装有相位。",
        "- 新 `stopGaitAtStand` 默认 false，I_Q 副本显式启用；历史模型行为保持。",
        "- 本次指令界限为 vx±0.55 m/s、vy±0.28 m/s、wz±0.65 rad/s、height±0.035 m、pitch±0.20 rad、roll±0.14 rad；这些是激励覆盖内的限制，没有辨识论文中的速度相关可行域。",
        "- 只验证当前平地 MJCF、固定步频及本次臂运动；其他步频、执行器语义、地形、载荷或真机需要重新验证。",
        "- 运行入口核对策略、配置、MJCF 和 MPC 任务散列；每次保存实际相位、指令、状态、EE目标/误差、求解耗时、任务和二进制散列。", "",
        "## 验证与追溯", "",
        "Python wire/benchmark contracts：21 passed。原生 policy-aware 模型/动力学/映射测试：22 passed。WbcBridgeCore 转换与相位测试：0 failed。包含启停解析预测与 C++ RK4 一致性、自动微分运行时相位门控、相位回绕与缺失相位拒绝。",
        "`models/` 为已废弃的首版拟合，启停残差离散积分存在不连续问题；正式模型为 `models_v2/`。原始 `prediction_validation*.json` 与 `models_v2_prediction_validation.json` 是过程记录，最终独立验证以 `independent_prediction_validation.json` 为准。",
        "`source/iq_response_model_fit_v2.py` 保存了散列与 selection.json 一致的精确拟合源码。原拟合的速度平方平滑项为 1e-8；冻结参数后，正式验证与 MPC 统一为 1e-6，没有重新拟合或选谐波。",
        "`mpc_preflight/gait/` 保留命令字段名错误的失败预检；`closed_loop/` 保留尚未展开相位的第一轮 18 条闭环。正式闭环以 `closed_loop_v2/` 为准；该修正只改变相位测量接入，没有再拟合模型或调整代价。",
        "重新评估冻结模型：", "", "```bash",
        "/opt/miniconda3/envs/base312/bin/python scripts/iq_response_model.py --evaluate-only \\",
        "  --data-root tmp/experiments/20260915_iq_identification/independent_holdout \\",
        "  --report-name independent_prediction_recheck.json", "```", "",
        "策略 SHA256：`60e70cc4ad2a885e0b469f7a39b11b7d851e46595b7a7382ee9eb10a205a14f0`。模型、原始数据和运行文件散列见各 manifest/receipt 以及 `handoff_manifest.json`。", ""]
    (root/"REPORT.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),layout="constrained")
    horizons=[.1,.3,.6,1.]
    for ax,c,unit in zip(axes,CHANNELS[3:],["mm","mrad","mrad"]):
        for model,label in [("F0","First order"),("F1_gait","First order + gait")]:
            ax.plot(horizons,[prediction["test"][model][str(h)][c]*1000 for h in horizons],"o-",label=label)
        ax.set(xlabel="Prediction horizon (s)",ylabel=f"RMSE ({unit})",title=c.capitalize());ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for ext in ["png","pdf"]: fig.savefig(root/f"prediction.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout="constrained")
    labels=["Hold","0.55 m curve","1.2 m curve"]
    for i,(controller,label) in enumerate([("ideal","Ideal"),("first_order","First order"),("gait","First order + gait")]):
        values=[r for r in groups if r["controller"]==controller]
        for ax,metric,scale in [(axes[0],"position_rmse_m",1000),(axes[1],"orientation_rmse_rad",1000)]:
            ax.bar(np.arange(3)+(i-1)*.24,[r[metric]*scale for r in values],.24,label=label)
            ax.set_xticks(range(3),labels,rotation=15)
        matching=[v for n,v in traces.items() if n.startswith(f"walking_curve_{controller}_s")]
        mean=np.mean([v["position_error"] for v in matching],axis=0)
        axes[2].plot(matching[0]["t"],mean*1000,label=label)
    axes[0].set(ylabel="EE position RMSE (mm)");axes[1].set(ylabel="EE orientation RMSE (mrad)")
    axes[2].set(xlabel="Time (s)",ylabel="EE position error (mm)",title="1.2 m curve")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=9)
    axes[2].grid(alpha=.2)
    for ext in ["png","pdf"]: fig.savefig(root/f"closed_loop.{ext}",dpi=180)
    plt.close(fig)
    print(root/"REPORT.md")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=str(ROOT/"tmp/experiments/20260915_iq_identification"))
    main(Path(p.parse_args().root).resolve())
