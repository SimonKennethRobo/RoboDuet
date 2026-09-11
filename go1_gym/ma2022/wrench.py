"""Vectorized equations (6)-(8), independent of IsaacGym."""

import torch


def world_to_body(q, vectors):
    """Rotate vectors by inverse xyzw quaternion (no change of wrench origin)."""
    xyz = -q[..., :3]
    uv = torch.cross(xyz, vectors, dim=-1)
    return vectors + 2 * (q[..., 3:] * uv + torch.cross(xyz, uv, dim=-1))


class WrenchSequence:
    def __init__(self, num_envs, cfg, device="cpu"):
        self.cfg, self.device = cfg, device
        self.low = torch.tensor(cfg.wrench_min, device=device)
        self.high = torch.tensor(cfg.wrench_max, device=device)
        self.scale = torch.tensor(cfg.wrench_scale, device=device)
        if not torch.all(self.low <= self.high) or not torch.all(self.scale > 0):
            raise ValueError("Invalid wrench bounds or scales")
        if not 0 <= cfg.beta_range[0] <= cfg.beta_range[1]:
            raise ValueError("beta_range must be nonnegative and ordered")
        self.knots = torch.zeros(num_envs, 3, 6, device=device)
        self.beta = torch.zeros(num_envs, 1, device=device)
        self.gains = torch.zeros(num_envs, 6, device=device)
        self.bias = torch.zeros(num_envs, 1, 6, device=device)
        self.noise_scale = torch.ones_like(self.bias)
        self.reset(torch.arange(num_envs, device=device))

    def _ball(self, count, radius):
        direction = torch.randn(count, 3, device=self.device)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return radius * direction * torch.rand(count, 1, device=self.device).pow(1 / 3)

    def reset(self, ids):
        n = len(ids)
        self.knots[ids] = self.low + torch.rand(n, 3, 6, device=self.device) * (self.high - self.low)
        lo, hi = self.cfg.beta_range
        self.beta[ids] = lo + torch.rand(n, 1, device=self.device) * (hi - lo)
        self.gains[ids, :3] = self._ball(n, self.cfg.force_gain_radius)
        self.gains[ids, 3:] = self._ball(n, self.cfg.torque_gain_radius)
        self.bias[ids] = torch.randn(n, 1, 6, device=self.device) * self.cfg.prediction_bias_std
        self.noise_scale[ids] = 1 + torch.randn(n, 1, 6, device=self.device) * self.cfg.prediction_scale_std

    def evaluate(self, times):
        t = torch.as_tensor(times, device=self.device, dtype=self.knots.dtype).view(1, -1, 1)
        a, b, c = self.knots.unbind(1)
        return (a[:, None] * ((t - 1) * (t - 2) / 2) -
                b[:, None] * (t * (t - 2)) + c[:, None] * (t * (t - 1) / 2))

    def advance(self, dt=0.02):
        shifted = self.evaluate([dt, 1 + dt])
        # Eq. (7) uses lower/upper magnitudes. For signed bounds, use abs
        # so its random-walk interval is ordered even for downward-only Fz.
        delta = -self.low.abs() + torch.rand_like(self.knots[:, 2]) * (
            self.low.abs() + self.high.abs())
        terminal = (self.knots[:, 2] + self.beta * delta).clamp(self.low, self.high)
        self.knots = torch.cat((shifted, terminal[:, None]), dim=1)

    def prediction(self, quat, noisy=False):
        world = self.evaluate(self.cfg.prediction_times)
        q = quat[:, None, :].expand(-1, world.shape[1], -1)
        body = torch.cat((world_to_body(q, world[..., :3]),
                          world_to_body(q, world[..., 3:])), dim=-1) * self.scale
        if noisy:
            body = body * self.noise_scale + self.bias + torch.randn_like(body) * self.cfg.prediction_noise_std
        return body.flatten(1)

    def disturbance(self, acceleration):
        std = torch.tensor(self.cfg.disturbance_std, device=self.device)
        return self.gains * acceleration + torch.randn_like(acceleration) * std

    def gain_target(self):
        return torch.stack((self.gains[:, :3].norm(dim=-1) / max(self.cfg.force_gain_radius, 1e-9),
                            self.gains[:, 3:].norm(dim=-1) / max(self.cfg.torque_gain_radius, 1e-9)), dim=-1)
