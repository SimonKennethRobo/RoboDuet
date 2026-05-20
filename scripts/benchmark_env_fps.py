#!/usr/bin/env python3
"""Benchmark RoboDuet env stepping FPS for several env counts.

The default path benchmarks the current stage-2 task shape with trajectory
tracking and dynamic gait enabled. Each env count is executed in a fresh child
process so global Cfg mutations and GPU allocations do not leak into the next
measurement.
"""

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ENV_NUMS = [512, 1024, 2048, 4096, 8192, 10240, 20480, 40960, 51200, 102400, 204800]

np = None
torch = None
HistoryWrapper = None
VelocityTrackingEasyEnv = None
Cfg = None
config_asset = None
config_go1 = None
config_wtw = None
global_switch = None


def load_isaac_modules():
    global np
    global torch
    global HistoryWrapper
    global VelocityTrackingEasyEnv
    global Cfg
    global config_asset
    global config_go1
    global config_wtw
    global global_switch

    if torch is not None:
        return

    import isaacgym

    assert isaacgym
    import numpy as _np
    import torch as _torch

    from go1_gym.envs.automatic import HistoryWrapper as _HistoryWrapper
    from go1_gym.envs.automatic import VelocityTrackingEasyEnv as _VelocityTrackingEasyEnv
    from go1_gym.envs.automatic.legged_robot_config import Cfg as _Cfg
    from go1_gym.envs.go1.asset_config import config_asset as _config_asset
    from go1_gym.envs.go1.go1_config import config_go1 as _config_go1
    from go1_gym.envs.go1.wtw_config import config_wtw as _config_wtw
    from go1_gym.utils import global_switch as _global_switch

    np = _np
    torch = _torch
    HistoryWrapper = _HistoryWrapper
    VelocityTrackingEasyEnv = _VelocityTrackingEasyEnv
    Cfg = _Cfg
    config_asset = _config_asset
    config_go1 = _config_go1
    config_wtw = _config_wtw
    global_switch = _global_switch


def configure_cfg(args):
    """Mirror the task-relevant auto_train.py config without wandb/runner setup."""

    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    Cfg.commands.distributional_commands = False
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False

    Cfg.control.control_type = "M"
    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions = False
    Cfg.commands.body_roll_range = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]

    Cfg.env.num_envs = args.num_envs
    Cfg.env.keep_arm_fixed = True
    Cfg.env.stage1_arm_curriculum = not args.no_stage1_arm_curriculum
    Cfg.env.stage1_arm_fixed_fraction = args.stage1_arm_fixed_fraction
    Cfg.env.stage1_arm_max_accel = args.stage1_arm_max_accel
    Cfg.env.stage1_arm_max_vel = args.stage1_arm_max_vel
    Cfg.env.stage1_arm_max_offset = args.stage1_arm_max_offset
    Cfg.env.stage1_arm_accel_resample_time_s = args.stage1_arm_accel_resample_time_s

    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
        Cfg.terrain.teleport_robots = False

    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63

    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.7 * Cfg.reward_scales.tracking_lin_vel
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.5 * Cfg.reward_scales.tracking_ang_vel
    Cfg.hybrid.reward_scales.arm_energy = -0.00004
    Cfg.reward_scales.loco_energy = -0.00004

    Cfg.reward_scales.jump = -0.00
    Cfg.rewards.terminal_body_height = 0.28
    Cfg.rewards.use_terminal_body_height = True

    Cfg.commands.T_force_range = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres = 0.3
    Cfg.domain_rand.max_force = 15
    Cfg.domain_rand.max_force_offset = 0.01
    if not args.randomize_materials:
        Cfg.domain_rand.randomize_friction = False
        Cfg.domain_rand.randomize_restitution = False

    Cfg.env.priv_observe_vel = False
    Cfg.commands.global_reference = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9

    Cfg.env.record_video = False
    Cfg.asset.render_sphere = False
    Cfg.hybrid.use_vision = False
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1
    Cfg.hybrid.reward_scales.arm_dof_vel = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = 5 * Cfg.reward_scales.action_smoothness_1
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = 5 * Cfg.reward_scales.action_smoothness_2

    Cfg.use_rot6d = args.use_rot6d
    if Cfg.use_rot6d:
        Cfg.env.num_observations += 3
        Cfg.arm.arm_num_observations += 3
        Cfg.arm.arm_num_obs_history = Cfg.arm.arm_num_observations * Cfg.arm.arm_num_observation_history
        Cfg.arm.arm_num_commands += 3
        Cfg.dog.dog_num_observations += 3
        Cfg.dog.dog_num_obs_history = Cfg.dog.dog_num_observations * Cfg.dog.dog_num_observation_history

    if args.traj_track:
        Cfg.arm.trajectory.enabled = True
        traj_window_dims = len(Cfg.arm.trajectory.window_offsets) * 9
        Cfg.arm.num_actions_arm_cd = Cfg.arm.num_actions_arm + 3
        Cfg.arm.arm_num_observations = 12 + 1 + 4 + 3 + 3 + 9 + 6 + traj_window_dims + 1
        Cfg.arm.arm_num_obs_history = Cfg.arm.arm_num_observations * Cfg.arm.arm_num_observation_history
        Cfg.dog.dog_num_observations += 9
        Cfg.dog.dog_num_obs_history = Cfg.dog.dog_num_observations * Cfg.dog.dog_num_observation_history
        Cfg.env.num_observations += 1 + 4 + 9 + 6 + traj_window_dims + 1
        Cfg.env.num_obs_history = Cfg.env.num_observation_history * Cfg.env.num_observations
        Cfg.hybrid.reward_scales.arm_manip_commands_tracking_combine = 0.0
        Cfg.hybrid.reward_scales.vis_manip_commands_tracking_lpy = 0.0
        Cfg.hybrid.reward_scales.vis_manip_commands_tracking_rpy = 0.0
        Cfg.hybrid.reward_scales.traj_track = 5.0
        Cfg.hybrid.reward_scales.trajectory_current_tracking = 1.0
        Cfg.hybrid.reward_scales.trajectory_completion_time = 0.5
        Cfg.hybrid.reward_scales.arm_delta_vel_cmd = -0.05
        Cfg.hybrid.reward_scales.ee_smoothness = -1e-4
    else:
        Cfg.arm.trajectory.enabled = False

    if args.dyna_gait:
        Cfg.commands.use_dynamic_gait = True
        Cfg.commands.gait_frequency_cmd_range = [
            args.dyna_gait_min_frequency,
            Cfg.commands.gait_frequency_cmd_range[1],
        ]
        Cfg.commands.limit_gait_frequency = Cfg.commands.gait_frequency_cmd_range
        Cfg.commands.limit_footswing_height = Cfg.commands.footswing_height_range
        Cfg.commands.limit_gait_duration = Cfg.commands.gait_duration_cmd_range
        Cfg.commands.limit_stance_width = Cfg.commands.stance_width_range
        Cfg.commands.limit_stance_length = Cfg.commands.stance_length_range

        num_new_gait_dims = 5
        Cfg.dog.dog_num_commands += num_new_gait_dims
        Cfg.dog.dog_num_observations += num_new_gait_dims
        Cfg.dog.dog_num_obs_history = Cfg.dog.dog_num_observations * Cfg.dog.dog_num_observation_history

        plan_action_dims = 7 + (3 if Cfg.arm.trajectory.enabled else 0)
        Cfg.arm.num_actions_arm_cd = Cfg.arm.num_actions_arm + plan_action_dims
        Cfg.arm.arm_num_observations += num_new_gait_dims
        Cfg.arm.arm_num_obs_history = Cfg.arm.arm_num_observations * Cfg.arm.arm_num_observation_history

        Cfg.env.num_observations += num_new_gait_dims + 2
        Cfg.env.observe_gait_commands = True

        Cfg.commands.num_bins_gait_frequency = 11
        Cfg.commands.num_bins_footswing_height = 5
        Cfg.commands.num_bins_gait_duration = 3
        Cfg.commands.num_bins_stance_width = 3
        Cfg.commands.num_bins_stance_length = 3
    else:
        Cfg.commands.use_dynamic_gait = False

    if args.robot == "go1":
        Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf"
    elif args.robot == "go2":
        Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf"

    if args.stage2:
        global_switch.pretrained_to_hybrid_start = 0
        global_switch.pretrained_to_hybrid_end = 1
    else:
        global_switch.pretrained_to_hybrid_start = args.steps + args.warmup_steps + 1000
        global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 1

    global_switch.count = 0
    global_switch.stage1_count = 0
    global_switch.switch_flag = False
    global_switch.init_sigmoid_lr()


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch_cuda_device_arg(device))


def cuda_device_index(sim_device):
    if not sim_device.startswith("cuda"):
        return None
    parts = sim_device.split(":", maxsplit=1)
    if len(parts) == 1 or parts[1] == "":
        return 0
    return int(parts[1])


def torch_cuda_device_arg(device):
    if not str(device).startswith("cuda"):
        return None
    return cuda_device_index(str(device))


def query_process_gpu_memory_mb(pid, gpu_index):
    if gpu_index is None:
        return None

    nvitop_memory = query_process_gpu_memory_mb_nvitop(pid)
    if nvitop_memory is not None:
        return nvitop_memory

    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        return None

    if proc.returncode != 0:
        return None

    total_mb = 0.0
    found = False
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            line_pid = int(fields[0])
            used_mb = float(fields[1])
        except ValueError:
            continue
        if line_pid == pid:
            total_mb += used_mb
            found = True
    return total_mb if found else None


def query_process_gpu_memory_mb_nvitop(pid):
    try:
        from nvitop import Device
    except Exception:
        return None

    try:
        devices = Device.all()
    except Exception:
        return None

    total_mb = 0.0
    found = False
    for device in devices:
        try:
            processes = device.processes()
        except Exception:
            continue
        if isinstance(processes, dict):
            process_items = processes.items()
        else:
            process_items = enumerate(processes)

        for process_pid, process in process_items:
            candidate_pid = getattr(process, "pid", process_pid)
            if callable(candidate_pid):
                try:
                    candidate_pid = candidate_pid()
                except Exception:
                    continue
            try:
                candidate_pid = int(candidate_pid)
            except (TypeError, ValueError):
                continue
            if candidate_pid != pid:
                continue

            memory_mb = extract_nvitop_memory_mb(process)
            if memory_mb is None:
                continue
            total_mb += memory_mb
            found = True

    return total_mb if found else None


def extract_nvitop_memory_mb(process):
    attr_names = (
        "gpu_memory",
        "gpu_memory_usage",
        "gpu_memory_used",
        "used_gpu_memory",
        "memory",
    )
    for attr_name in attr_names:
        memory_mb = normalize_memory_mb(getattr(process, attr_name, None))
        if memory_mb is not None:
            return memory_mb

    snapshot_fn = getattr(process, "as_snapshot", None)
    if callable(snapshot_fn):
        try:
            snapshot = snapshot_fn()
        except Exception:
            snapshot = None
        if snapshot is not None:
            for attr_name in attr_names:
                memory_mb = normalize_memory_mb(getattr(snapshot, attr_name, None))
                if memory_mb is not None:
                    return memory_mb
            if isinstance(snapshot, dict):
                for attr_name in attr_names:
                    memory_mb = normalize_memory_mb(snapshot.get(attr_name))
                    if memory_mb is not None:
                        return memory_mb

    human_fn = getattr(process, "gpu_memory_human", None)
    memory_mb = normalize_memory_mb(human_fn)
    if memory_mb is not None:
        return memory_mb

    return None


def normalize_memory_mb(value):
    if value is None:
        return None
    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    if isinstance(value, (int, float)):
        # nvitop may report bytes in some APIs and MiB in others.
        if value > 1024**2:
            return float(value) / 1024**2
        return float(value)
    if isinstance(value, str):
        match = re.search(r"([0-9.]+)\s*([KMGT]?i?B|[KMGT]?B)?", value.strip(), re.IGNORECASE)
        if match is None:
            return None
        number = float(match.group(1))
        unit = (match.group(2) or "MiB").lower()
        if unit in ("b",):
            return number / 1024**2
        if unit in ("kb", "kib"):
            return number / 1024
        if unit in ("mb", "mib"):
            return number
        if unit in ("gb", "gib"):
            return number * 1024
        if unit in ("tb", "tib"):
            return number * 1024**2
    return None


class GPUMemorySampler:
    def __init__(self, pid, gpu_index, interval_s):
        self.pid = pid
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mb = None
        self.last_mb = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.gpu_index is None or self.interval_s <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4.0))
        self.sample_once()

    def sample_once(self):
        used_mb = query_process_gpu_memory_mb(self.pid, self.gpu_index)
        if used_mb is None:
            return
        self.last_mb = used_mb
        if self.peak_mb is None or used_mb > self.peak_mb:
            self.peak_mb = used_mb

    def _run(self):
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)


def torch_memory_stats_mb(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    cuda_device = torch_cuda_device_arg(device)
    try:
        return {
            "torch_allocated_mb": torch.cuda.memory_allocated(cuda_device) / 1024**2,
            "torch_reserved_mb": torch.cuda.memory_reserved(cuda_device) / 1024**2,
            "torch_max_allocated_mb": torch.cuda.max_memory_allocated(cuda_device) / 1024**2,
            "torch_max_reserved_mb": torch.cuda.max_memory_reserved(cuda_device) / 1024**2,
        }
    except Exception:
        return {}


def make_actions(env):
    dog_actions = torch.empty(
        env.num_envs,
        env.num_actions_loco,
        dtype=torch.float,
        device=env.device,
        requires_grad=False,
    )
    arm_policy_actions = torch.empty(
        env.num_envs,
        env.cfg.arm.num_actions_arm_cd,
        dtype=torch.float,
        device=env.device,
        requires_grad=False,
    )
    return dog_actions, arm_policy_actions


def step_random_actions(env, dog_actions, arm_policy_actions, include_observations):
    dog_actions.uniform_(-1.0, 1.0)
    arm_policy_actions.uniform_(-1.0, 1.0)

    if global_switch.switch_open and env.num_plan_actions > 0:
        env.plan(arm_policy_actions[:, -env.num_plan_actions :])
        arm_actions = arm_policy_actions[:, : env.num_actions_arm]
    else:
        arm_actions = arm_policy_actions[:, : env.num_actions_arm]

    env.step(dog_actions, arm_actions)

    if include_observations:
        env.get_dog_observations()
        if global_switch.switch_open:
            env.get_arm_observations()


def run_single(args):
    pid = os.getpid()
    gpu_index = cuda_device_index(args.sim_device)
    memory_sampler = GPUMemorySampler(pid, gpu_index, args.memory_sample_interval)
    memory_sampler.start()

    load_isaac_modules()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if gpu_index is not None:
            try:
                torch.cuda.reset_peak_memory_stats(gpu_index)
            except Exception:
                pass

    configure_cfg(args)

    env = VelocityTrackingEasyEnv(
        sim_device=args.sim_device,
        headless=True,
        num_envs=args.num_envs,
        cfg=Cfg,
        graphics_device_id=args.graphics_device_id,
    )
    env = HistoryWrapper(env)

    if args.stage2:
        global_switch.count = global_switch.pretrained_to_hybrid_end
        global_switch.open_switch()

    env.reset()
    dog_actions, arm_policy_actions = make_actions(env)

    for _ in range(args.warmup_steps):
        step_random_actions(env, dog_actions, arm_policy_actions, args.include_observations)
    synchronize(env.device)
    memory_sampler.sample_once()

    start = time.perf_counter()
    for _ in range(args.steps):
        step_random_actions(env, dog_actions, arm_policy_actions, args.include_observations)
    synchronize(env.device)
    elapsed = time.perf_counter() - start
    memory_sampler.stop()

    fps = args.num_envs * args.steps / elapsed
    memory_stats = torch_memory_stats_mb(env.device)
    result = {
        "num_envs": args.num_envs,
        "fps": fps,
        "steps": args.steps,
        "seconds": elapsed,
        "stage": "stage2" if args.stage2 else "stage1",
        "traj_track": args.traj_track,
        "dyna_gait": args.dyna_gait,
        "include_observations": args.include_observations,
        "sim_device": args.sim_device,
        "gpu_index": gpu_index,
        "proc_gpu_memory_mb": memory_sampler.last_mb,
        "proc_gpu_peak_memory_mb": memory_sampler.peak_mb,
        **memory_stats,
    }

    proc_mem = "n/a" if result["proc_gpu_memory_mb"] is None else f"{result['proc_gpu_memory_mb']:.0f} MB"
    proc_peak = "n/a" if result["proc_gpu_peak_memory_mb"] is None else f"{result['proc_gpu_peak_memory_mb']:.0f} MB"
    print(
        f"{args.num_envs:>5} envs | {fps:,.0f} FPS | {elapsed:.3f}s "
        f"| proc mem {proc_mem}, peak {proc_peak} "
        f"({result['stage']}, traj_track={args.traj_track}, dyna_gait={args.dyna_gait})",
        flush=True,
    )
    print("BENCH_JSON " + json.dumps(result, sort_keys=True), flush=True)

    del dog_actions, arm_policy_actions, env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def child_args_for(parent_args, num_envs):
    child = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--num_envs",
        str(num_envs),
        "--sim_device",
        parent_args.sim_device,
        "--robot",
        parent_args.robot,
        "--warmup_steps",
        str(parent_args.warmup_steps),
        "--steps",
        str(parent_args.steps),
        "--seed",
        str(parent_args.seed),
        "--dyna_gait_min_frequency",
        str(parent_args.dyna_gait_min_frequency),
        "--memory_sample_interval",
        str(parent_args.memory_sample_interval),
    ]
    if parent_args.graphics_device_id is not None:
        child += ["--graphics_device_id", str(parent_args.graphics_device_id)]
    if parent_args.stage2:
        child.append("--stage2")
    else:
        child.append("--stage1")
    if parent_args.traj_track:
        child.append("--traj_track")
    else:
        child.append("--no_traj_track")
    if parent_args.dyna_gait:
        child.append("--dyna_gait")
    else:
        child.append("--no_dyna_gait")
    if parent_args.use_rot6d:
        child.append("--use_rot6d")
    if parent_args.include_observations:
        child.append("--include_observations")
    else:
        child.append("--no_include_observations")
    if parent_args.no_stage1_arm_curriculum:
        child.append("--no_stage1_arm_curriculum")
    if parent_args.randomize_materials:
        child.append("--randomize_materials")
    child += [
        "--stage1_arm_fixed_fraction",
        str(parent_args.stage1_arm_fixed_fraction),
        "--stage1_arm_max_accel",
        str(parent_args.stage1_arm_max_accel),
        "--stage1_arm_max_vel",
        str(parent_args.stage1_arm_max_vel),
        "--stage1_arm_max_offset",
        str(parent_args.stage1_arm_max_offset),
        "--stage1_arm_accel_resample_time_s",
        str(parent_args.stage1_arm_accel_resample_time_s),
    ]
    return child


def run_all(args):
    results = []
    env = os.environ.copy()
    env.setdefault("WANDB_SILENT", "true")

    print(
        "Benchmark config: "
        f"stage={'stage2' if args.stage2 else 'stage1'}, "
        f"traj_track={args.traj_track}, dyna_gait={args.dyna_gait}, "
        f"steps={args.steps}, warmup={args.warmup_steps}, device={args.sim_device}",
        flush=True,
    )

    for num_envs in args.env_nums:
        proc = subprocess.run(
            child_args_for(args, num_envs),
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")

        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("BENCH_JSON "):
                result = json.loads(line[len("BENCH_JSON ") :])
        if proc.returncode != 0:
            print(f"{num_envs} envs failed with return code {proc.returncode}", flush=True)
            results.append({"num_envs": num_envs, "error": f"return code {proc.returncode}"})
        elif result is not None:
            results.append(result)
        else:
            print(f"{num_envs} envs did not emit BENCH_JSON", flush=True)
            results.append({"num_envs": num_envs, "error": "missing BENCH_JSON"})

    print(
        "\n| env_num | FPS | seconds | proc_gpu_mb | proc_peak_gpu_mb | "
        "torch_alloc_mb | torch_reserved_mb | torch_peak_alloc_mb | torch_peak_reserved_mb | status |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    for result in results:
        if "error" in result:
            print(f"| {result['num_envs']} | - | - | - | - | - | - | - | - | {result['error']} |")
        else:
            proc_mem = result.get("proc_gpu_memory_mb")
            proc_peak = result.get("proc_gpu_peak_memory_mb")
            torch_alloc = result.get("torch_allocated_mb")
            torch_reserved = result.get("torch_reserved_mb")
            torch_peak_alloc = result.get("torch_max_allocated_mb")
            torch_peak_reserved = result.get("torch_max_reserved_mb")
            print(
                f"| {result['num_envs']} | {result['fps']:.0f} | {result['seconds']:.3f} | "
                f"{format_optional_mb(proc_mem)} | "
                f"{format_optional_mb(proc_peak)} | "
                f"{format_optional_mb(torch_alloc)} | "
                f"{format_optional_mb(torch_reserved)} | "
                f"{format_optional_mb(torch_peak_alloc)} | "
                f"{format_optional_mb(torch_peak_reserved)} | ok |"
            )


def format_optional_mb(value):
    return "-" if value is None else f"{value:.0f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RoboDuet env FPS.")
    parser.add_argument("--single", action="store_true", help="Internal mode: benchmark one env count.")
    parser.add_argument("--env_nums", nargs="+", type=int, default=DEFAULT_ENV_NUMS)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--memory_sample_interval", type=float, default=0.5)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])

    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--stage2", action="store_true", default=True, help="Benchmark arm+loco stage2 path.")
    stage.add_argument("--stage1", dest="stage2", action="store_false", help="Benchmark stage1 loco-only path.")

    parser.add_argument("--traj_track", dest="traj_track", action="store_true", default=True)
    parser.add_argument("--no_traj_track", dest="traj_track", action="store_false")
    parser.add_argument("--dyna_gait", dest="dyna_gait", action="store_true", default=True)
    parser.add_argument("--no_dyna_gait", dest="dyna_gait", action="store_false")
    parser.add_argument("--include_observations", dest="include_observations", action="store_true", default=True)
    parser.add_argument("--no_include_observations", dest="include_observations", action="store_false")
    parser.add_argument("--use_rot6d", action="store_true", default=False)
    parser.add_argument("--no_stage1_arm_curriculum", action="store_true", default=False)
    parser.add_argument("--randomize_materials", action="store_true", default=False)
    parser.add_argument("--stage1_arm_fixed_fraction", type=float, default=0.1)
    parser.add_argument("--stage1_arm_max_accel", type=float, default=2.0)
    parser.add_argument("--stage1_arm_max_vel", type=float, default=1.0)
    parser.add_argument("--stage1_arm_max_offset", type=float, default=0.35)
    parser.add_argument("--stage1_arm_accel_resample_time_s", type=float, default=0.5)
    parser.add_argument("--dyna_gait_min_frequency", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.single:
        run_single(parsed)
    else:
        run_all(parsed)
