"""Train Ma et al. (2022) teacher or recurrent student on bare Go2."""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import pickle
import random
import sys

import isaacgym  # Must precede torch imports.
import numpy as np
import torch

from go1_gym.envs.config import apply_config_snapshot, cfg_to_dict
from go1_gym.envs.config.ma2022 import MaTrainingConfig, build_ma_config
from go1_gym.ma2022.env import MaLocomotionEnv
from go1_gym.ma2022.models import Teacher, Student
from go1_gym.ma2022.logging import WandbLogger
from go1_gym.ma2022.video import TrainingVideo
from go1_gym.ma2022.training import (
    export_student, load_checkpoint, save_checkpoint, student_iteration, teacher_iteration,
)
from go1_gym.utils.global_switch import global_switch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--teacher", type=Path, help="Required teacher checkpoint for student distillation")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20000, help="Total target iteration count")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--terrain", choices=("plane", "trimesh"), default=None,
                        help="Override terrain; default is rough ground for a new teacher")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=Path)
    parser.add_argument("--no_video", action="store_true", help="Disable training video recording")
    parser.add_argument("--video_interval", type=int, default=500, help="Iterations between clips")
    parser.add_argument("--video_length", type=float, default=10., help="Clip length in simulation seconds")
    parser.add_argument("--video_stride", type=int, default=2, help="Policy steps between frames (default 25 fps)")
    parser.add_argument("--run_name", help="W&B run name; defaults to the log directory name")
    parser.add_argument("--wandb_project", default="roboduet")
    parser.add_argument("--wandb_entity", default="simon00715", help="W&B team/user (same default as auto_train.py)")
    parser.add_argument("--notes", default="", help="W&B run notes")
    parser.add_argument("--offline", action="store_true", help="Save W&B data locally for later sync")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B (takes precedence over --offline)")
    parser.add_argument("--rollout_steps", type=int, help="Override Python recipe for bounded validation")
    parser.add_argument("--no_wrench_prediction", action="store_true",
                        help="Ma Fig. 5 ablation: zero the wrench-prediction observation (new teacher only)")
    args = parser.parse_args()
    if args.video_interval < 1 or args.video_stride < 1 or not 0 < args.video_length < float("inf"):
        parser.error("Video interval, stride and finite duration must be positive")
    if args.num_envs < 1 or args.iterations < 1 or (args.rollout_steps is not None and args.rollout_steps < 1):
        parser.error("Environment, iteration and rollout counts must be positive")
    if args.stage == "student" and args.teacher is None:
        parser.error("--stage student requires --teacher")
    if args.stage == "teacher" and args.teacher is not None:
        parser.error("--teacher is only valid for student distillation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    recipe = MaTrainingConfig()
    cfg = build_ma_config(args.num_envs)
    teacher_checkpoint = load_checkpoint(args.teacher) if args.teacher else None
    resume = load_checkpoint(args.resume) if args.resume else None
    if teacher_checkpoint and teacher_checkpoint["stage"] != "teacher":
        parser.error("--teacher must point to a teacher checkpoint")
    if resume and resume["stage"] != args.stage:
        parser.error("Resume stage differs from --stage")
    source = resume or teacher_checkpoint
    if source:
        apply_config_snapshot(cfg, source["env_cfg"], strict=False)
        recipe = MaTrainingConfig(**source["recipe"])
        cfg.env.num_envs = args.num_envs
    if args.terrain:
        cfg.terrain.mesh_type = args.terrain
    if args.rollout_steps:
        recipe.rollout_steps = args.rollout_steps
    if args.no_wrench_prediction:
        if source:
            parser.error("--no_wrench_prediction only applies to a new teacher; the checkpoint recipe decides")
        recipe.observe_wrench_prediction = False
    cfg.env.record_video = not args.no_video
    cfg.env.recording_width_px = 640
    cfg.env.recording_height_px = 480
    # No learned-arm curriculum or response-consistency ramp in this task.
    global_switch.switch_flag = False
    global_switch.count = 0
    global_switch.pretrained_to_wbc_start = 10**12
    global_switch.pretrained_to_wbc_end = 10**12 + 1
    global_switch.init_sigmoid_lr()
    log_dir = args.log_dir or Path("runs/ma2022") / f"{datetime.now():%Y%m%d_%H%M%S}_{args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Configuration is checkpoint-owned after training; the editable recipe
    # remains Python. No experiment config is loaded from JSON.
    with (log_dir / "parameters.pkl").open("wb") as stream:
        pickle.dump(dict(Cfg=cfg_to_dict(cfg), MaTrainingConfig=asdict(recipe), args=vars(args)), stream)
    env = MaLocomotionEnv(cfg, recipe, args.sim_device, args.headless)
    wandb_logger = None
    recorder = None
    try:
        if cfg.env.record_video:
            recorder = TrainingVideo(log_dir, args.stage, env.dt, interval=args.video_interval,
                                     seconds=args.video_length, stride=args.video_stride)
            env.video_recorder = recorder
        dims = env.observation_dims
        model = (Teacher(dims, recipe) if args.stage == "teacher" else Student(dims, recipe)).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe.learning_rate)
        start = 0
        if resume:
            if resume["dims"] != dims:
                raise ValueError("Resume observation layout mismatch")
            model.load_state_dict(resume["model"], strict=True)
            optimizer.load_state_dict(resume["optimizer"])
            start = resume["iteration"]
        teacher = None
        if teacher_checkpoint:
            if teacher_checkpoint["dims"] != dims:
                raise ValueError("Teacher observation layout mismatch")
            teacher_recipe = MaTrainingConfig(**teacher_checkpoint["recipe"])
            teacher_settings, student_settings = asdict(teacher_recipe), asdict(recipe)
            teacher_settings.pop("rollout_steps")
            student_settings.pop("rollout_steps")
            if teacher_settings != student_settings:
                raise ValueError("Teacher and student checkpoint recipes differ")
            teacher = Teacher(dims, teacher_recipe).to(env.device)
            teacher.load_state_dict(teacher_checkpoint["model"], strict=True)
            teacher.eval().requires_grad_(False)
        obs = env.observations()
        states = (torch.zeros(args.num_envs, recipe.hidden_dim, device=env.device),
                  torch.zeros(args.num_envs, recipe.hidden_dim, device=env.device))
        reset = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        wandb_logger = WandbLogger(args, cfg, recipe, log_dir)
        print(f"Ma2022 {args.stage}: dims={dims}, actions=16, log_dir={log_dir}", flush=True)
        for iteration in range(start + 1, args.iterations + 1):
            global_switch.count = iteration
            env.set_reward_curriculum(iteration - 1)
            if recorder is not None:
                recorder.start(iteration, first=iteration == start + 1)
            if args.stage == "teacher":
                obs, metrics = teacher_iteration(env, model, optimizer, recipe, obs)
            else:
                obs, states, reset, metrics = student_iteration(
                    env, model, teacher, optimizer, recipe, obs, states, reset)
            line = f"iteration={iteration} " + " ".join(f"{k}={v:.6g}" for k, v in metrics.items())
            print(line, flush=True)
            if recorder is not None and iteration == args.iterations:
                recorder.finish()
            wandb_logger.log(iteration, metrics, num_envs=args.num_envs,
                             rollout_steps=recipe.rollout_steps,
                             learning_rate=optimizer.param_groups[0]["lr"],
                             videos=recorder.pop_completed() if recorder is not None else ())
            try:
                with (log_dir / "metrics.log").open("a") as stream:
                    stream.write(line + "\n")
            except OSError as error:
                if iteration == start + 1 or iteration % 100 == 0:
                    print(f"Skipping noncritical metrics write: {error}", flush=True)
            if iteration % recipe.save_interval == 0 or iteration == args.iterations:
                save_checkpoint(log_dir / f"{args.stage}_{iteration:06d}.pt", model, optimizer,
                                iteration, args.stage, env, recipe)
        if args.stage == "student":
            export_student(model, log_dir / "student_jit.pt")
    finally:
        try:
            if recorder is not None:
                recorder.finish()
            if wandb_logger is not None:
                wandb_logger.finish(exit_code=1 if sys.exc_info()[0] else 0)
        finally:
            env.close()


if __name__ == "__main__":
    main()
