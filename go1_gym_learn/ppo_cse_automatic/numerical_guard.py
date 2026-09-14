"""Opt-in fail-fast diagnostics at policy and PPO numerical boundaries."""
from pathlib import Path
import torch


class NumericalGuard:
    def __init__(self, model, optimizer, directory, context=None):
        self.model, self.optimizer = model, optimizer
        self.directory = Path(directory)
        self.context = context
        self.phase = 'initialization'
        self.iteration = None

    def check(self, stage, **values):
        values = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
        if values and not torch.stack([torch.isfinite(v).all() for v in values.values()]).all().item():
            self.fail(stage, values)

    def check_distribution(self, mean, std):
        self.check('actor_distribution', mean=mean, std=std)
        if not (std > 0).all().item():
            self.fail('nonpositive_std', dict(mean=mean, std=std))

    def check_gradients(self):
        self.check('gradients', **{k: p.grad for k, p in self.model.named_parameters() if p.grad is not None})

    def check_parameters(self):
        self.check('parameters', **dict(self.model.named_parameters()))

    def fail(self, stage, values):
        # The failure path alone copies data/weights to CPU; no repair or update
        # is attempted for invalid model data.
        details, samples, row_ids = {}, {}, None
        for key, value in values.items():
            value = value.detach()
            bad = ~torch.isfinite(value)
            details[key] = dict(shape=list(value.shape), nonfinite=int(bad.sum().item()))
            if value.ndim > 0:
                rows = bad.reshape(value.shape[0], -1).any(dim=1).nonzero().flatten()[:8]
                if value.ndim >= 2 and rows.numel() and row_ids is None:
                    row_ids = rows
                samples[key] = (value if value.ndim == 1 and value.numel() <= 4096
                                else value[rows if rows.numel() else slice(0, 8)]).cpu()
            else:
                samples[key] = value.cpu()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / 'policy-fault.pt'
        payload = dict(phase=self.phase, stage=stage, iteration=self.iteration,
                       details=details, samples=samples,
                       model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                       gradients={k: p.grad.detach().cpu() for k, p in self.model.named_parameters() if p.grad is not None},
                       optimizer=self.optimizer.state_dict())
        if self.context is not None and self.phase == 'rollout' and row_ids is not None:
            payload['physics_context'] = self.context(row_ids)
        if self.phase == 'rollout' and row_ids is not None:
            payload['actor_inputs'] = {k: v[row_ids].detach().cpu()
                                       for k, v in getattr(self, 'last_inputs', {}).items()}
        torch.save(payload, path)
        raise FloatingPointError(f'Numerical fault at {self.phase}/{stage}, iteration={self.iteration}; saved {path}')
