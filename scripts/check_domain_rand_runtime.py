"""Eight environments and 32 zero-action steps; no training or checkpoint writes."""
import argparse
from argparse import Namespace

import isaacgym  # Must precede torch.
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils.global_switch import global_switch


def main():
    parser = argparse.ArgumentParser(description="Bounded GPU smoke for DR, terrain and reset integration.")
    parser.add_argument("--mode", choices=("sim2real", "benchmark", "none"), default="sim2real")
    mode = parser.parse_args().mode
    cfg = build_roboduet_config(Namespace(num_envs=8, robot="go2", domain_rand_mode=mode, dyna_gait=True))
    cfg.terrain.num_rows = 3
    cfg.terrain.num_cols = 6
    cfg.terrain.border_size = 2
    cfg.domain_rand.randomize_rigids_after_start = True
    cfg.domain_rand.dog_obs_latency_steps_range = [1, 1]
    cfg.domain_rand.mount_tf_buckets = 2
    global_switch.switch_flag = False
    env = WBCEnv(sim_device="cuda:0", headless=True, cfg=cfg)
    try:
        env.stage1_arm_play_intensity = 1.0
        wrapped = HistoryWrapper(env)
        wrapped.reset()
        creation_com = env.com_displacements.clone()
        dog_actions = torch.zeros(8, cfg.dog.num_actions_loco, device="cuda:0")
        arm_actions = torch.zeros(8, cfg.arm.num_actions_arm_cd, device="cuda:0")
        for step in range(32):
            wrapped.step(dog_actions, arm_actions)
            env.get_dog_observations()  # Repeated read must not advance latency.
            if step in (5, 15):
                env.reset_idx(torch.tensor([0, 2], device="cuda:0"))
        assert torch.isfinite(env.root_states).all()
        assert torch.equal(env.com_displacements, creation_com)
        if mode != "none":
            assert env.stage1_ee_payload_com.abs().max().item() > 0
        assert env.dog_obs_latency.delays.max().item() == (1 if mode == "sim2real" else 0)
        for i in range(8):
            props = env.gym.get_actor_rigid_body_properties(env.envs[i], env.actor_handles[i])
            assert abs(props[env.base_mass_body_index].mass - env.default_body_mass - float(env.payloads[i])) < 1e-4
        print("DR_SMOKE_OK", mode, "obs", cfg.dog.dog_num_observations,
              "columns", env.terrain_types.tolist(), "delay", env.dog_obs_latency.delays.tolist())
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
