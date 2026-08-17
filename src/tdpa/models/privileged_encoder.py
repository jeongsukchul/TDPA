from __future__ import annotations

import torch
from torch import nn

from tdpa.models.history_encoder import _masked_gru_final


class PrivilegedEncoder(nn.Module):
    """Training-only teacher, anchored through the response objective."""

    def __init__(self, privileged_dim: int = 16, latent_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.temporal = nn.GRU(privileged_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(
        self, privileged_history: torch.Tensor, history_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return a training-only privileged latent with shape ``[B, latent_dim]``."""
        if privileged_history.ndim != 3:
            raise ValueError("privileged_history must have shape [B,H,D]")
        values = privileged_history
        if history_mask is not None:
            values = values * history_mask.unsqueeze(-1).to(values.dtype)
        return self.head(_masked_gru_final(self.temporal, values, history_mask))
