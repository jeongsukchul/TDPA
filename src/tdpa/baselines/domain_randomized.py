from __future__ import annotations

import torch
from torch import nn


class DomainRandomizedPolicy(nn.Module):
    """Task policy baseline trained across train-support physics without context."""

    def __init__(self, proprio_dim: int = 10, action_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.network(proprio)

