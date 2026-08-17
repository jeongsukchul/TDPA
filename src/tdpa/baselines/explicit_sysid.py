from __future__ import annotations

import torch
from torch import nn


class ExplicitSystemIdentifier(nn.Module):
    """Estimate interpretable mass/friction, then feed the same adapter contract."""

    def __init__(self, proprio_dim: int = 10, action_dim: int = 7, hidden_dim: int = 64) -> None:
        super().__init__()
        self.temporal = nn.GRU(proprio_dim + action_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        proprio_history: torch.Tensor,
        action_history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = torch.cat([proprio_history, action_history], dim=-1)
        if history_mask is not None:
            inputs = inputs * history_mask.unsqueeze(-1).to(inputs.dtype)
        sequence, _ = self.temporal(inputs)
        raw = self.head(sequence[:, -1])
        # Positive physics estimates; probe supervision never updates a shared encoder.
        return torch.nn.functional.softplus(raw) + 1e-4
