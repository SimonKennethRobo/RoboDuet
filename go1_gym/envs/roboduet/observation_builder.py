"""Small helpers for policy observation assembly and validation."""

import torch


class ObservationBuilder:
    def __init__(self, env, name, expected_dim):
        self.env = env
        self.name = name
        self.expected_dim = expected_dim
        self.terms = []

    def add(self, *terms):
        for term in terms:
            if term is None:
                continue
            if term.shape[-1] == 0:
                continue
            self.terms.append(term)

    def build(self, clip=True):
        if self.terms:
            obs = torch.cat(self.terms, dim=-1)
        else:
            obs = torch.empty(self.env.num_envs, 0, dtype=torch.float, device=self.env.device)

        if obs.shape[1] != self.expected_dim:
            raise AssertionError(
                f"{self.name} num_observations ({self.expected_dim}) != "
                f"the number of observations ({obs.shape[1]})"
            )

        if clip:
            clip_obs = self.env.cfg.normalization.clip_observations
            obs = torch.clip(obs, -clip_obs, clip_obs)
        return obs


def clip_observation(env, obs):
    clip_obs = env.cfg.normalization.clip_observations
    return torch.clip(obs, -clip_obs, clip_obs)
