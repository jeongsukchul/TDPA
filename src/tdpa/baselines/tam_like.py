from __future__ import annotations

import torch
from torch import nn


class TAMLike(nn.Module):
    """Closest local reference: bounded low-level tracking-residual correction.

    This is not labeled as the official TAM implementation. See the baseline
    audit for the deviation.
    """

    def __init__(self, action_dim: int = 4, tracking_dim: int = 3, limit: float = 0.15) -> None:
        super().__init__()
        self.limit = limit
        self.network = nn.Sequential(
            nn.Linear(action_dim + tracking_dim, 64), nn.SiLU(), nn.Linear(64, action_dim)
        )

    def forward(self, nominal_action: torch.Tensor, tracking_residual: torch.Tensor) -> torch.Tensor:
        residual = self.limit * torch.tanh(
            self.network(torch.cat([nominal_action, tracking_residual], dim=-1))
        )
        return torch.clamp(nominal_action + residual, -1.0, 1.0)

