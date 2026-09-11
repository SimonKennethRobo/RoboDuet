"""Interactive Ma teacher/student playback against the synthetic wrench task."""

import argparse

import isaacgym  # Import before torch.
import torch

from go1_gym.envs.config import apply_config_snapshot
from go1_gym.envs.config.ma2022 import MaTrainingConfig, build_ma_config
from go1_gym.ma2022.env import MaLocomotionEnv
from go1_gym.ma2022.models import Teacher, Student
from go1_gym.ma2022.training import load_checkpoint
from go1_gym.utils.global_switch import global_switch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--sim_device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    recipe = MaTrainingConfig(**checkpoint["recipe"])
    cfg = build_ma_config(1)
    apply_config_snapshot(cfg, checkpoint["env_cfg"], strict=False)
    cfg.env.num_envs = 1
    cfg.env.record_video = False
    global_switch.switch_flag = False
    global_switch.count = 0
    global_switch.pretrained_to_wbc_start = 10**12
    global_switch.pretrained_to_wbc_end = 10**12 + 1
    global_switch.init_sigmoid_lr()
    env = MaLocomotionEnv(cfg, recipe, args.sim_device, headless=False)
    try:
        if env.observation_dims != checkpoint["dims"]:
            raise ValueError("Checkpoint observation layout mismatch")
        student = checkpoint["stage"] == "student"
        model = (Student(env.observation_dims, recipe) if student else
                 Teacher(env.observation_dims, recipe)).to(env.device).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        h_w = torch.zeros(1, recipe.hidden_dim, device=env.device)
        h_b = torch.zeros_like(h_w)
        reset = torch.ones(1, dtype=torch.bool, device=env.device)
        obs = env.observations()
        with torch.no_grad():
            while True:
                if student:
                    output = model(obs["student_proprio"], obs["student_wrench"], obs["student_scan"],
                                   h_w, h_b, reset)
                    action, h_w, h_b = output[:3]
                else:
                    action = model(obs)[0]
                obs, _, reset, _ = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
