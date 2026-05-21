
import isaacgym
assert isaacgym
import torch
import argparse

from go1_gym.envs.roboduet.stage_schedule import StageSchedule, apply_hybrid_reward_settings
from go1_gym.envs.roboduet.wbc_env_config import RoboDuetCfg as Cfg, configure_task_from_args

import wandb
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import shutil
import pickle

from go1_gym.envs.roboduet import HistoryWrapper, WBCEnv

from go1_gym_learn.ppo_cse_unified import Runner
from go1_gym_learn.ppo_cse_unified.ppo import UnifiedPPO_Args
from go1_gym_learn.ppo_cse_unified import UnifiedRunnerArgs
from go1_gym_learn.ppo_cse_unified.unified2head_ac import Unified2AC_Args

from go1_gym.utils import format_code, set_seed, global_switch
os.environ["WANDB_SILENT"] = "true"

def configure_train_stage(args):
    schedule = StageSchedule(
        args.train_stage,
        args.num_learning_iterations,
        default_switch_iteration=10000,
        debug=args.debug,
    )
    schedule.configure(global_switch)

    if args.debug:
        UnifiedRunnerArgs.save_interval = 2
        UnifiedRunnerArgs.save_video_interval = 10


def unified_reward_scales_wrapper(self):
    def get_reward_scales():
        return self.hybrid_reward_scales

    return get_reward_scales

def train_go1(headless=True):

    if args.debug:
        mode = "disabled"
        args.num_envs = 12
    else:
        mode = "online"

        if args.offline:
            mode = "offline"

    if args.no_wandb:
        mode = "disabled"

    if args.resume:
        args.tags.append("resume")

    args.seed = set_seed(args.seed)
    args.tags.append(f"seed{args.seed}")

    configure_task_from_args(Cfg, args, traj_track_reward_scale=1.0)
    Unified2AC_Args.num_actions_arm = Cfg.arm.num_actions_arm_cd
    configure_train_stage(args)

    global_switch.init_sigmoid_lr()
    # global_switch.init_linear_lr()

    if args.train_stage == "stage2":
        apply_hybrid_reward_settings(Cfg)

    # if args.headless:
    #     UnifiedRunnerArgs.log_video = False

    now = datetime.now()
    stem = Path(__file__).stem
    wandb.init(entity="simon00715",
               project="roboduet",
               group=args.run_name,
               mode=mode,
               notes=args.notes,
               name=f'{now.strftime("%Y-%m-%d")}/{stem}/{now.strftime("%H%M%S.%f")}',
               tags=args.tags,
               dir=f"{MINI_GYM_ROOT_DIR}")

    args.log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs/{args.run_name}", wandb.run.name)
    args.log_dir += f'_seed{args.seed}'
    if not args.debug:
        os.makedirs(osp.join(args.log_dir, "checkpoints_unified"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "deploy_model"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "scripts"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "videos"), exist_ok=True)
        os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/unified_train.py", f"{args.log_dir}/scripts/unified_train.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/legged_robot.py", f"{args.log_dir}/scripts/legged_robot.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/legged_robot_config.py", f"{args.log_dir}/scripts/legged_robot_config.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/__init__.py", f"{args.log_dir}/scripts/env__init__.py")
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/wbc_env.py",
            f"{args.log_dir}/scripts/wbc_env.py",
        )
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/observation_builder.py", f"{args.log_dir}/scripts/observation_builder.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/trajectory_geometry.py", f"{args.log_dir}/scripts/trajectory_geometry.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/stage_schedule.py", f"{args.log_dir}/scripts/stage_schedule.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/wbc_env_config.py", f"{args.log_dir}/scripts/wbc_env_config.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet/asset_config.py", f"{args.log_dir}/scripts/asset_config.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/go1/go1_config.py", f"{args.log_dir}/scripts/go1_config.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/go1/wtw_config.py", f"{args.log_dir}/scripts/wtw_config.py")

        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_unified/__init__.py", f"{args.log_dir}/scripts/ppo_cse_unified__init__.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_unified/unified2head_ac.py", f"{args.log_dir}/scripts/unified2head_ac.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_unified/ppo.py", f"{args.log_dir}/scripts/ppo.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_unified/rollout_storage.py", f"{args.log_dir}/scripts/rollout_storage.py")


        wandb.run.log_code(f"{args.log_dir}/scripts")

        temp_dict = {"Cfg": vars(Cfg), "RunnerArgs": vars(UnifiedRunnerArgs), "Unified2AC_Args": vars(Unified2AC_Args), "PPO_Args": vars(UnifiedPPO_Args),}

        with open(f"{args.log_dir}/params.txt", "w", encoding="utf-8") as f:
            format_temp_dict = format_code(str(temp_dict))
            f.write(format_temp_dict)

        with open(osp.join(args.log_dir, "parameters.pkl"), 'wb') as f:
            pickle.dump(temp_dict, f)
        wandb.save(osp.join(args.log_dir, "parameters.pkl"), policy="now")

        wandb.log({
            "Global_Switch/start": global_switch.pretrained_to_hybrid_start,
            "Global_Switch/end": global_switch.pretrained_to_hybrid_end,
            }, step=0)

    env = WBCEnv(
        sim_device=args.sim_device,
        headless=args.headless,
        cfg=Cfg,
        graphics_device_id=args.graphics_device_id,
    )
    env = HistoryWrapper(env)

    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(env, device=f"cuda:{gpu_id}", run_name=args.run_name, resume=args.resume, log_dir=args.log_dir, debug=args.debug)
    runner.learn(num_learning_iterations=args.num_learning_iterations, init_at_random_ep_len=True, eval_freq=args.eval_freq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--graphics_device_id', type=int, default=None)
    parser.add_argument('--num_learning_iterations', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--num_envs', type=int, default=2048)
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tags', nargs='+', default=[])
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--robot', type=str, default="go1", choices=["go1", "go2"])
    parser.add_argument('--train_stage', type=str, default="two_stage", choices=["stage1", "stage2", "two_stage"])
    parser.add_argument('--use_rot6d', action='store_true', default=False)
    parser.add_argument('--dyna_gait', action='store_true', default=False)
    parser.add_argument('--traj_track', action='store_true', default=False)

    args = parser.parse_args()

    train_go1(args)
