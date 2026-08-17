from __future__ import annotations

import torch
from torch import nn


class ResponseEncoder(nn.Module):
    """Fixed response-sensitive target transform with no collapse-prone bias."""

    def __init__(
        self, response_dim: int = 12, embedding_dim: int = 32, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        del hidden_dim
        self.response_dim = response_dim
        rows = response_dim * 3
        row = torch.arange(1, rows + 1, dtype=torch.float32).unsqueeze(1)
        column = torch.arange(1, embedding_dim + 1, dtype=torch.float32).unsqueeze(0)
        projection = torch.sin(row * column * 0.173) + torch.cos(row * column * 0.117)
        projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("projection", projection)

    def forward(
        self, response_sequence: torch.Tensor, response_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if response_sequence.ndim != 3:
            raise ValueError("response_sequence must have shape [B,F,D]")
        if response_sequence.shape[-1] != self.response_dim:
            raise ValueError("Unexpected response feature dimension")
        if response_mask is None:
            response_mask = torch.ones(
                response_sequence.shape[:2], dtype=torch.bool, device=response_sequence.device
            )
        if not bool(response_mask.any(dim=1).all()):
            raise ValueError("Every response target needs at least one paired timestep")
        weights = response_mask.unsqueeze(-1).to(response_sequence.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (response_sequence * weights).sum(dim=1) / count
        centered = (response_sequence - mean.unsqueeze(1)) * weights
        deviation = torch.sqrt(centered.square().sum(dim=1) / count + 1e-8)
        positions = torch.arange(response_mask.shape[1], device=response_mask.device).expand_as(
            response_mask
        )
        last_index = torch.where(response_mask, positions, -1).max(dim=1).values
        last = response_sequence[
            torch.arange(response_sequence.shape[0], device=response_sequence.device), last_index
        ]
        summary = torch.cat([mean, last, deviation], dim=-1)
        return summary @ self.projection.to(summary.dtype)
