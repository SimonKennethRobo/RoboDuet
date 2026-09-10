"""Per-environment transport delay on a fixed-width signal.

Used for the sensing latency between a measurement being taken on the robot
and the policy seeing it: IMU / encoder sampling, the driver's own buffering,
the network hop and the inference call all sit between the two, and none of
them exist in the simulator, where ``get_dog_observations`` reads the state
the physics step just wrote.

Kept free of IsaacGym so the ring-buffer indexing can be unit tested on CPU.
"""

import torch


class LatencyBuffer:
    """Delay each environment's signal by its own integer number of pushes.

    Integer steps, deliberately.  A fractional delay has to interpolate
    between two ring slots, and a two-tap interpolation is a low-pass filter:
    it would attenuate exactly the high-frequency content of ``dof_vel`` and
    ``base_ang_vel`` that the policy uses to feel the gait, so the delay would
    arrive bundled with a smoothing the real robot does not apply.  The
    quantisation error that buys is at most half a policy step (10 ms), which
    is smaller than the spread between robots that the range randomisation is
    there to cover in the first place.

    Args:
        num_envs: number of environments.
        width: width of the delayed signal.
        max_steps: largest delay, in pushes, that ``delays`` may hold.
        device / dtype: as for the signal being delayed.
    """

    def __init__(self, num_envs, width, max_steps, device, dtype=torch.float):
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}")
        self.num_envs = int(num_envs)
        self.width = int(width)
        self.max_steps = int(max_steps)
        self.device = device
        # One slot for the freshest sample plus one per step of delay.
        self.capacity = self.max_steps + 1
        self.buffer = torch.zeros(
            self.capacity, self.num_envs, self.width, dtype=dtype, device=device
        )
        self.delays = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        # Index of the newest sample.  Starts at the last slot so that the
        # first push lands on slot 0 and the ring fills in order.
        self.head = self.capacity - 1
        self._primed = False

    def sample_delays(self, env_ids, low, high):
        """Draw a fresh per-episode delay for ``env_ids`` from [low, high] steps.

        Both bounds are inclusive and are rounded to whole steps: the range is
        a physical latency spread across robots, so the caller passes it in
        steps already scaled by the domain-randomisation curriculum.
        """
        if len(env_ids) == 0:
            return
        lo = max(0, int(round(float(low))))
        hi = min(self.max_steps, int(round(float(high))))
        if hi < lo:
            lo, hi = hi, lo
        self.delays[env_ids] = torch.randint(
            lo, hi + 1, (len(env_ids),), device=self.device, dtype=torch.long
        )

    def reset_idx(self, env_ids, values):
        """Fill every slot of ``env_ids`` with their current measurement.

        Called when an episode restarts.  Without it a freshly reset robot
        would spend its first ``delay`` steps observing the *previous*
        episode's final state -- a discontinuity no real latency produces,
        and one that lands precisely where the reference model is being
        aligned to the measured state.
        """
        if len(env_ids) == 0:
            return
        self.buffer[:, env_ids] = values[env_ids].unsqueeze(0).to(self.buffer.dtype)

    def push(self, values):
        """Record this step's measurement as the newest sample."""
        if values.shape != (self.num_envs, self.width):
            raise ValueError(
                f"expected values of shape {(self.num_envs, self.width)}, got {tuple(values.shape)}"
            )
        if not self._primed:
            # The ring starts at zero, which is not a state the robot was ever
            # in; before the first push there is no history to serve.
            self.buffer[:] = values.unsqueeze(0).to(self.buffer.dtype)
            self._primed = True
        self.head = (self.head + 1) % self.capacity
        self.buffer[self.head] = values.to(self.buffer.dtype)

    def read(self, jitter_steps=0):
        """Return, per environment, the sample ``delays`` pushes ago.

        ``jitter_steps`` adds an independent per-step, per-env integer offset
        drawn from [-jitter, +jitter], clamped into the buffer.  It models the
        scheduling jitter of a non-realtime sensing loop; the constant part of
        the latency stays in ``delays``, where the domain curriculum can scale
        it.
        """
        delays = self.delays
        if jitter_steps:
            jitter = torch.randint(
                -int(jitter_steps),
                int(jitter_steps) + 1,
                (self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
            delays = (delays + jitter).clamp_(0, self.max_steps)
        index = ((self.head - delays) % self.capacity).view(1, -1, 1).expand(1, -1, self.width)
        return torch.gather(self.buffer, 0, index).squeeze(0)
