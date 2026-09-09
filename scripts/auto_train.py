import isaacgym

assert isaacgym
import argparse
import os
import os.path as osp
import pickle
import shutil
from datetime import datetime

import wandb
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.config import ARM_ACTION_MODES, build_roboduet_config, cfg_to_dict
from go1_gym.envs.roboduet.utils import StageSchedule, apply_wbc_reward_settings
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils import format_code, global_switch, set_seed
from go1_gym.utils.wandb_config import build_wandb_config
from go1_gym_learn.ppo_cse_automatic import ArmRunnerArgs, DogRunnerArgs, Runner, RunnerArgs
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmAC_Args
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogAC_Args
from go1_gym_learn.ppo_cse_automatic.ppo import PPO_Args

os.environ["WANDB_SILENT"] = "true"


_DOG_COMMAND_LIMIT_FIELDS = (
    "limit_vel_x",
    "limit_vel_y",
    "limit_vel_yaw",
    "limit_body_pitch",
    "limit_body_roll",
    "limit_body_height",
    "limit_gait_frequency",
    "limit_footswing_height",
    "limit_stance_width",
    "limit_stance_length",
    "limit_gait_duration",
)


def _logdir_from_dog_ckpt_path(ckpt_path):
    if ckpt_path is None:
        return None
    ckpt_path = osp.abspath(ckpt_path)
    parent = osp.basename(osp.dirname(ckpt_path))
    if parent == "checkpoints_dog":
        return osp.dirname(osp.dirname(ckpt_path))
    if osp.isdir(ckpt_path):
        return ckpt_path
    return osp.dirname(ckpt_path)


def apply_dog_checkpoint_command_limits(cfg, ckpt_path):
    logdir = _logdir_from_dog_ckpt_path(ckpt_path)
    if logdir is None:
        return
    params_path = osp.join(logdir, "parameters.pkl")
    if not osp.exists(params_path):
        print(f"[warn] dog parameters.pkl not found at {params_path}; using current command limits.", flush=True)
        return
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    dog_cfg = params.get("Cfg") if isinstance(params, dict) else None
    if isinstance(dog_cfg, dict):
        dog_commands = dog_cfg.get("commands")
    else:
        dog_commands = getattr(dog_cfg, "commands", None)
    if dog_commands is None:
        print(f"[warn] dog parameters.pkl has no Cfg.commands; using current command limits.", flush=True)
        return
    copied = []
    print(f"Loaded dog policy parameters from {params_path}", flush=True)
    print("Dog command limits applied to stage2:", flush=True)
    for name in _DOG_COMMAND_LIMIT_FIELDS:
        if isinstance(dog_commands, dict):
            value = dog_commands.get(name)
        else:
            value = getattr(dog_commands, name, None)
        if value is not None:
            value = list(value) if isinstance(value, tuple) else value
            setattr(cfg.commands, name, value)
            copied.append(name)
            print(f"  {name}: {value}", flush=True)
    if not copied:
        print("  [warn] no dog command limit fields were found; using current command limits.", flush=True)


def _cfg_snapshot_with_command_limits(cfg):
    return cfg_to_dict(cfg)


def configure_train_stage(args, cfg):
    schedule = StageSchedule(
        args.train_stage,
        args.num_learning_iterations,
        default_switch_iteration=2000 if args.resume else 8000,
        stage1_arm_ramp_iterations=cfg.env.stage1_arm_ramp_iterations,
        debug=args.debug,
    )
    schedule.configure(global_switch)

    if args.debug:
        RunnerArgs.save_interval = 2
        RunnerArgs.save_video_interval = 10


def main(args):

    if args.debug:
        mode = "disabled"
        args.num_envs = 4
        args.video = True
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

    cfg = build_roboduet_config(args, debug=args.debug)
    if cfg.terrain.reset_mode == "fixed_mixture":
        print("[reset mixture] " + str({k: v for k, v in cfg_to_dict(cfg.terrain).items()
                                       if k.startswith("reset_mix")}), flush=True)
        print(f"[push] max angular velocity per world axis: {cfg.domain_rand.max_push_ang_vel} rad/s", flush=True)
    cfg.env.arm_policy_enabled = args.train_stage != "stage1"
    cfg.env.record_video = args.video
    if not cfg.env.record_video:
        RunnerArgs.log_video = False
    RunnerArgs.num_steps_per_env = args.num_steps_per_env
    PPO_Args.num_mini_batches = args.num_mini_batches

    stage2_freeze_loco_policy = not args.stage2_unfreeze_loco_policy
    DogRunnerArgs.ckpt_path = args.stage1_ckpt_path
    if args.train_stage != "stage1" and DogRunnerArgs.ckpt_path is None and stage2_freeze_loco_policy:
        if stage2_freeze_loco_policy:
            print(
                "[warn] --stage2_freeze_loco_policy requires --stage1_ckpt_path for stage2 training; "
                "forcing --stage2_unfreeze_loco_policy.",
                flush=True,
            )
        stage2_freeze_loco_policy = False
    apply_dog_checkpoint_command_limits(cfg, DogRunnerArgs.ckpt_path)
    DogRunnerArgs.stage2_freeze_loco_policy = stage2_freeze_loco_policy
    DogRunnerArgs.stage2_loco_learning_rate = args.stage2_loco_learning_rate
    ArmRunnerArgs.ckpt_path = args.stage2_ckpt_path
    print("-" * 20 + " Configured Train Stage " + "-" * 20)
    print(f"DogRunnerArgs: {vars(DogRunnerArgs)}")
    print("-" * 10)
    print(f"ArmRunnerArgs: {vars(ArmRunnerArgs)}")
    print("-" * 50)

    configure_train_stage(args, cfg)

    global_switch.init_sigmoid_lr()
    # global_switch.init_linear_lr()

    if args.train_stage == "stage2":
        apply_wbc_reward_settings(cfg)

    now = datetime.now()
    wandb_config = build_wandb_config(
        args=args,
        Cfg=cfg,
        RunnerArgs=RunnerArgs,
        ArmRunnerArgs=ArmRunnerArgs,
        DogRunnerArgs=DogRunnerArgs,
        ArmAC_Args=ArmAC_Args,
        DogAC_Args=DogAC_Args,
        PPO_Args=PPO_Args,
        GlobalSwitch={
            "pretrained_to_wbc_start": global_switch.pretrained_to_wbc_start,
            "pretrained_to_wbc_end": global_switch.pretrained_to_wbc_end,
            "stage1_arm_ramp_iterations": getattr(global_switch, "stage1_arm_ramp_iterations", None),
        },
    )
    wandb.init(
        entity="simon00715",
        project="roboduet",
        group=args.run_name,
        mode=mode,
        notes=args.notes,
        name=f"{now.strftime('%Y-%m-%d')}/{args.run_name}_{now.strftime('%H%M%S')}",
        tags=args.tags,
        dir=f"{MINI_GYM_ROOT_DIR}",
        config=wandb_config,
    )

    if args.debug:
        args.log_dir = osp.join(
            f"{MINI_GYM_ROOT_DIR}/runs",
            f"{now.strftime('%Y-%m-%d')}/debug_{args.run_name}_{now.strftime('%H%M%S')}",
        )
    else:
        args.log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs", wandb.run.name)
    print(f"Logging to {args.log_dir}")
    # args.log_dir += f"_seed{args.seed}"

    os.makedirs(osp.join(args.log_dir, "checkpoints_arm"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "checkpoints_dog"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "videos"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "deploy_model"), exist_ok=True)
    os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

    if not args.debug:
        os.makedirs(osp.join(args.log_dir, "scripts"), exist_ok=True)
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/auto_train.py", f"{args.log_dir}/scripts/auto_train.py")
        for root, dirs, files in os.walk(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet"):
            rel_root = osp.relpath(root, f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/roboduet")
            target_root = (
                osp.join(args.log_dir, "scripts", rel_root) if rel_root != "." else osp.join(args.log_dir, "scripts")
            )
            os.makedirs(target_root, exist_ok=True)
            for filename in files:
                if filename.endswith(".py"):
                    shutil.copyfile(osp.join(root, filename), osp.join(target_root, filename))
        shutil.copytree(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/config",
            f"{args.log_dir}/scripts/config",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py", f"{args.log_dir}/scripts/arm_ac.py"
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/dog_ac.py", f"{args.log_dir}/scripts/dog_ac.py"
        )

        temp_dict = {
            "Cfg": _cfg_snapshot_with_command_limits(cfg),
            "RunnerArgs": vars(RunnerArgs),
            "ArmAC_Args": vars(ArmAC_Args),
            "DogAC_Args": vars(DogAC_Args),
            "PPO_Args": vars(PPO_Args),
        }

        with open(f"{args.log_dir}/params.txt", "w", encoding="utf-8") as f:
            format_temp_dict = format_code(str(temp_dict))
            f.write(format_temp_dict)

        with open(osp.join(args.log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump(temp_dict, f)
        wandb.save(osp.join(args.log_dir, "parameters.pkl"), policy="now")

        wandb.log(
            {
                "Global_Switch/start": global_switch.pretrained_to_wbc_start,
                "Global_Switch/end": global_switch.pretrained_to_wbc_end,
            },
            step=0,
        )

    env = WBCEnv(
        sim_device=args.sim_device,
        headless=args.headless,
        cfg=cfg,
        graphics_device_id=args.graphics_device_id,
    )
    env = HistoryWrapper(env)
    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(
        env, device=f"cuda:{gpu_id}", run_name=args.run_name, resume=args.resume, log_dir=args.log_dir, debug=args.debug
    )
    runner.learn(
        num_learning_iterations=args.num_learning_iterations, init_at_random_ep_len=True, eval_freq=args.eval_freq
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--num_learning_iterations", type=int, default=100000)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")  # for two_stage
    parser.add_argument("--tags", nargs="+", default=[])
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--robot", type=str, default="go2_x5", choices=["go1", "go2", "go2_x5"])
    parser.add_argument("--video", action="store_true", default=False)

    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_steps_per_env", type=int, default=RunnerArgs.num_steps_per_env)
    parser.add_argument("--num_mini_batches", type=int, default=PPO_Args.num_mini_batches)

    parser.add_argument("--train_stage", type=str, default="two_stage", choices=["stage1", "stage2", "two_stage"])
    stage2_loco_group = parser.add_mutually_exclusive_group()
    stage2_loco_group.add_argument("--stage2_unfreeze_loco_policy", action="store_true", default=False)
    parser.add_argument("--stage2_loco_learning_rate", type=float, default=None)
    parser.add_argument("--stage1_ckpt_path", type=str, default=None)
    parser.add_argument("--stage2_ckpt_path", type=str, default=None)

    parser.add_argument("--dyna_gait", action="store_true", default=False)
    parser.add_argument(
        "--goal_reaching",
        action="store_true",
        default=False,
        help="Train the 12D whole-body upper policy on static world-frame 6D goals.",
    )
    parser.add_argument(
        "--traj_tracking",
        action="store_true",
        default=False,
        help="Train the whole-body upper policy to track a moving SE(3) trajectory "
        "(implies --goal_reaching; sets wbc.goal_reaching.target_mode='trajectory').",
    )

    parser.add_argument(
        "--arm_action_mode",
        type=str,
        default=None,
        choices=list(ARM_ACTION_MODES),
        help="What the upper policy's 6 arm action dims mean. 'ik_residual': "
        "DLS-IK tracks the target and the policy adds a per-joint delta_q. "
        "'ik_waypoint': the policy outputs an intermediate EE waypoint "
        "(dpos+drot in the base frame) and IK solves for that. 'end_to_end': "
        "no IK -- the policy outputs arm joint position targets directly. "
        "All three share the same action/obs layout. Omit to use the "
        "arm.action_mode value set in config/wbc.py.",
    )

    parser.add_argument(
        "--no_reach_table",
        action="store_true",
        default=False,
        help="Ablation: use the scalar reach_radius sphere for rho/v_ff instead of "
        "the M2 direction-dependent reachability table.",
    )

    parser.add_argument('--raibert_exp', action='store_true', default=False,
                        help='Use exponential Raibert reward with v3-stage2 weights (0.4 / 0.2).')
    args = parser.parse_args()

    main(args)
