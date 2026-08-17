from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tdpa.models.history_encoder import _masked_gru_final


class ResponsePredictor(nn.Module):
    """Action-conditioned predictor; future action is never hidden in the latent."""

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_encoder = nn.GRU(action_dim, hidden_dim, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        dynamics_latent: torch.Tensor,
        future_action_sequence: torch.Tensor,
        future_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict future physical-response embedding."""
        if dynamics_latent.ndim != 2 or future_action_sequence.ndim != 3:
            raise ValueError("Expected latent [B,D] and future actions [B,F,A]")
        actions = future_action_sequence
        if future_action_mask is not None:
            actions = actions * future_action_mask.unsqueeze(-1).to(actions.dtype)
        action_feature = _masked_gru_final(self.action_encoder, actions, future_action_mask)
        return self.predictor(torch.cat([dynamics_latent, action_feature], dim=-1))


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(prediction, dim=-1), F.normalize(target, dim=-1))
