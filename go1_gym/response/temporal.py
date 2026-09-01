"""R7.3 -- the temporal encoder that consumes the observation history.

Lives here rather than beside ``DogActorCritic`` for the same reason the rest of
this package does: ``go1_gym_learn.ppo_cse_automatic.__init__`` imports the
IsaacGym environment, so anything defined under it cannot be unit-tested without
starting a simulator.  ``dog_ac.py`` imports the class from here.
"""

from __future__ import annotations

import torch.nn as nn


class EncoderDefaults:
    """Fallback knobs, mirrored from ``DogAC_Args`` so this module stays
    importable without the learner package."""

    temporal_encoder = "flat"
    tcn_channels = 128
    tcn_kernel_size = 3
    tcn_dilations = [1, 2, 4, 8, 16]


class TemporalEncoder(nn.Module):
    """Consumes the flattened observation history; emits one feature vector.

    The four structural constraints R7.3 asks for, so that swapping ``flat`` for
    ``tcn`` later is a genuine drop-in, all live here or are enforced by this
    signature:

    1. The history buffer stays contiguous per time step, oldest first --
       enforced by ``HistoryWrapper`` and pinned by a test, because a "helpful"
       interleaved layout would silently make the reshape below wrong rather
       than make it fail.
    2. The choice is one switch, ``self._args.temporal_encoder``.
    3. **The reshape belongs to the encoder.**  The public signature is always
       ``(B, T*C) -> (B, out_dim)``, so the deployment contract --
       ``torch.jit.script`` over a single flat tensor -- holds for both modes
       and the export path never learns which one is in use.
    4. ``T`` and ``C`` are derived from the config, never hard-coded.
    """

    def __init__(self, num_obs, num_history_steps, out_dim, activation, extra_dim=0,
                 args=None):
        super().__init__()
        self._args = args if args is not None else EncoderDefaults
        self.num_obs = int(num_obs)
        self.num_history_steps = int(num_history_steps)
        self.extra_dim = int(extra_dim)
        self.flat_dim = self.num_obs * self.num_history_steps + self.extra_dim
        self.mode = self._args.temporal_encoder
        self.out_dim = int(out_dim)

        # Both submodules exist in both modes: torch.jit.script compiles every
        # branch of forward() regardless of which one runs, so a head that only
        # exists for the tcn path makes the flat path unscriptable.
        self.head = nn.Identity()
        self.receptive_field = self.num_history_steps
        if self.mode == "flat":
            self.body = nn.Sequential(nn.Linear(self.flat_dim, self.out_dim), activation)
        elif self.mode == "tcn":
            if self.extra_dim:
                raise ValueError(
                    "the TCN encoder needs a pure (T, C) history; it cannot be used "
                    "with the adaptation module's concatenated privileged vector"
                )
            layers = []
            channels = int(self._args.tcn_channels)
            kernel = int(self._args.tcn_kernel_size)
            in_channels = self.num_obs
            receptive = 1
            for dilation in self._args.tcn_dilations:
                pad = int(dilation) * (kernel - 1)
                # Causal padding as a LAYER, not as logic in forward().  An
                # isinstance() check inside forward is not scriptable, and
                # torch.jit.script over the actor is the deployment contract --
                # so a padding decision taken at call time would break export
                # for the tcn mode only, which is exactly the kind of breakage
                # that surfaces months later.
                layers.append(nn.ConstantPad1d((pad, 0), 0.0))
                layers.append(
                    nn.Conv1d(in_channels, channels, kernel, dilation=int(dilation))
                )
                layers.append(activation)
                receptive += (kernel - 1) * int(dilation)
                in_channels = channels
            if receptive < self.num_history_steps:
                raise ValueError(
                    f"TCN receptive field {receptive} < history {self.num_history_steps}; "
                    "widen tcn_dilations"
                )
            self.body = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Linear(channels, self.out_dim), activation)
            self.receptive_field = receptive
        else:
            raise ValueError(f"unknown temporal_encoder {self.mode!r}")

    def forward(self, x):
        if self.mode == "flat":
            return self.body(x)
        # (B, T*C) -> (B, C, T): Conv1d wants channels first, and the history is
        # laid out oldest-step-first with each step contiguous.
        batch = x.shape[0]
        sequence = x.view(batch, self.num_history_steps, self.num_obs).transpose(1, 2)
        sequence = self.body(sequence)
        return self.head(sequence[:, :, -1])
