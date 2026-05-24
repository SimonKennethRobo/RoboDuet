import time
import isaacgym
import torch
from isaacgym.torch_utils import *
from go1_gym.envs import *
from go1_gym.envs.roboduet.wbc_env_wrapper import KeyboardWrapper
from go1_gym.envs.roboduet.wbc_env_config import configure_privileged_obs_dims
from scripts.load_policy import load_dog_policy, load_arm_policy, load_env
import argparse

logdir = "runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0


def play_go1(args):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd, logdir, ckpt_id

    logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)

    from go1_gym.utils.global_switch import global_switch

    stage1_only = bool(getattr(args, "stage1_only", False))
    if stage1_only:
        global_switch.switch_flag = False
        global_switch.count = 0
        global_switch.stage1_arm_ramp_iterations = 1
        global_switch.stage1_count = 1
        global_switch.pretrained_to_hybrid_start = args.num_eval_steps + 1
        global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 1
    else:
        global_switch.open_switch()

    env, cfg = load_env(
        logdir,
        wrapper=KeyboardWrapper,
        headless=args.headless,
        device=args.sim_device,
        robot=getattr(args, "robot", None),
    )
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = None if stage1_only else load_arm_policy(logdir, ckpt_id, cfg)
    configure_privileged_obs_dims(cfg)

    env.env.enable_viewer_sync = True

    num_eval_steps = args.num_eval_steps

    ''' press 'F' to fixed camera'''
    # cam_pos = gymapi.Vec3(4, 3, 2)
    # cam_target = gymapi.Vec3(-4, -3, 0)
    # env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

    obs = env.reset()

    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
    env.commands_dog[:, 2] = yaw_vel_cmd
    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd
    if hasattr(env.env, "_reset_keyboard_play_commands"):
        env.env._reset_keyboard_play_commands()

    for i in (range(num_eval_steps)):

        with torch.no_grad():
            if arm_policy is None:
                actions_arm = env.arm_fake_actions
            else:
                obs = env.get_arm_observations()
                actions_arm = arm_policy(obs).to(env.env.device)
                env.plan(actions_arm[..., -2:])

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs).to(env.env.device)
        if arm_policy is None:
            ret = env.step(actions_dog, actions_arm)
        else:
            ret = env.step(actions_dog, actions_arm[..., :-2])



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--ckptid', type=int, default=40000)
    parser.add_argument('--robot', type=str, default="go2", choices=["go1", "go2"])
    parser.add_argument('--num_eval_steps', type=int, default=30000)
    parser.add_argument(
        '--stage1_only',
        action='store_true',
        default=False,
        help='Run pure stage1 play: keep global switch closed and do not load/call the arm policy.',
    )

    args = parser.parse_args()
    play_go1(args)
