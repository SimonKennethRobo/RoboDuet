"""Figure 4 teacher and two-stream recurrent student; explicit hidden state."""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, output_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(),
                         nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, output_dim))


class Teacher(nn.Module):
    def __init__(self, dims, cfg):
        super().__init__()
        h, e = cfg.hidden_dim, cfg.embedding_dim
        self.wrench_encoder = mlp(dims["wrench"], e, h)
        self.scan_encoder = mlp(dims["scan"], e, h)
        self.privileged_encoder = mlp(dims["privileged"], e, h)
        self.actor = mlp(dims["proprio"] + 3*e, 16, h)
        self.critic = mlp(dims["proprio"] + 3*e, 1, h)
        self.log_std = nn.Parameter(torch.zeros(16))

    def forward(self, obs: Dict[str, torch.Tensor]):
        wrench = self.wrench_encoder(obs["wrench"])
        belief = torch.cat((self.scan_encoder(obs["scan"]), self.privileged_encoder(obs["privileged"])), dim=-1)
        features = torch.cat((obs["proprio"], wrench, belief), dim=-1)
        return self.actor(features), self.critic(features).squeeze(-1), wrench, belief


class Student(nn.Module):
    def __init__(self, dims, cfg):
        super().__init__()
        h, e = cfg.hidden_dim, cfg.embedding_dim
        self.hidden_dim = h
        self.scan_encoder = mlp(dims["scan"], e, h)
        self.wrench_rnn = nn.GRUCell(dims["proprio"] + dims["wrench"], h)
        self.belief_rnn = nn.GRUCell(dims["proprio"] + e, h)
        self.wrench_embedding = mlp(h, e, h)
        self.belief_embedding = mlp(h, 2*e, h)
        self.actor = mlp(dims["proprio"] + 3*e, 16, h)
        # All decoders use belief history only, never the wrench RNN.
        self.privileged_decoder = mlp(h, dims["privileged"], h)
        self.scan_decoder = mlp(h, dims["scan"], h)
        self.wrench_decoder = mlp(h, 6, h)
        self.gain_decoder = mlp(h, 2, h)

    def forward(self, proprio: torch.Tensor, wrench: torch.Tensor, scan: torch.Tensor,
                wrench_state: torch.Tensor, belief_state: torch.Tensor,
                reset: torch.Tensor):
        """Reset flags apply BEFORE this observation, one flag per env."""
        keep = (~reset.to(torch.bool)).to(proprio.dtype).unsqueeze(-1)
        wrench_state = self.wrench_rnn(torch.cat((proprio, wrench), dim=-1), wrench_state * keep)
        belief_state = self.belief_rnn(torch.cat((proprio, self.scan_encoder(scan)), dim=-1), belief_state * keep)
        w = self.wrench_embedding(wrench_state)
        b = self.belief_embedding(belief_state)
        action = self.actor(torch.cat((proprio, w, b), dim=-1))
        return (action, wrench_state, belief_state, w, b,
                self.privileged_decoder(belief_state), self.scan_decoder(belief_state),
                self.wrench_decoder(belief_state), self.gain_decoder(belief_state))


class DeployedStudent(nn.Module):
    """TorchScript interface: available observations + two states -> action + states."""
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, proprio: torch.Tensor, wrench: torch.Tensor, scan: torch.Tensor,
                wrench_state: torch.Tensor, belief_state: torch.Tensor, reset: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.student(proprio, wrench, scan, wrench_state, belief_state, reset)
        return out[0], out[1], out[2]


def distillation_losses(student_output, teacher_output, obs):
    action, _, _, w, b, privileged, scan, applied, gains = student_output
    teacher_action, _, teacher_w, teacher_b = teacher_output
    return {
        "action": F.mse_loss(action, teacher_action.detach()),
        "embedding": F.mse_loss(w, teacher_w.detach()) + F.mse_loss(b, teacher_b.detach()),
        "privileged": F.mse_loss(privileged, obs["privileged"]),
        "scan": F.mse_loss(scan, obs["scan"]),
        "w1": F.mse_loss(applied, obs["applied_wrench"]),
        "w2": F.mse_loss(gains, obs["gain"]),
    }
