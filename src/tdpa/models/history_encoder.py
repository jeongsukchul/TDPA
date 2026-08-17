from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence


def _last_valid(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence[:, -1]
    if mask.shape != sequence.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match sequence {sequence.shape[:2]}")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every history must contain at least one valid timestep")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = torch.where(mask, positions, -1).max(dim=1).values
    return sequence[torch.arange(sequence.shape[0], device=sequence.device), indices]


def _masked_gru_final(
    gru: nn.GRU, sequence: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is None:
        _, hidden = gru(sequence)
        return hidden[-1]
    if mask.shape != sequence.shape[:2]:
        raise ValueError("Mask and sequence dimensions do not match")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sequence needs at least one valid timestep")
    compact = [values[valid] for values, valid in zip(sequence, mask)]
    packed = pack_sequence(compact, enforce_sorted=False)
    _, hidden = gru(packed)
    return hidden[-1]


class HistoryEncoder(nn.Module):
    """Deployment dynamics encoder over RGB-D, proprioception, and past action."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        image_channels: int = 4,
        use_rgbd: bool = True,
        mask_goal_channel: bool = True,
    ) -> None:
        super().__init__()
        self.use_rgbd = use_rgbd
        self.mask_goal_channel = mask_goal_channel
        self.image_feature_dim = hidden_dim // 2
        if use_rgbd:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(24, self.image_feature_dim),
                nn.LayerNorm(self.image_feature_dim),
            )
        else:
            self.image_encoder = None
            self.image_feature_dim = 0
        self.sensor_projection = nn.Sequential(
            nn.Linear(proprio_dim + action_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.temporal = nn.GRU(hidden_dim + self.image_feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(
        self,
        rgbd_history: torch.Tensor | None,
        proprio_history: torch.Tensor,
        action_history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return dynamics latent ``z`` with shape ``[B, latent_dim]``."""
        if proprio_history.ndim != 3 or action_history.ndim != 3:
            raise ValueError("proprio_history and action_history must be [B,H,D]")
        if proprio_history.shape[:2] != action_history.shape[:2]:
            raise ValueError("History modalities must share B,H dimensions")
        sensor = self.sensor_projection(torch.cat([proprio_history, action_history], dim=-1))
        features = [sensor]
        if self.use_rgbd:
            if rgbd_history is None or rgbd_history.ndim != 5:
                raise ValueError("RGB-D history must be [B,H,C,W,W] when enabled")
            if rgbd_history.shape[:2] != proprio_history.shape[:2]:
                raise ValueError("RGB-D and proprio histories must be aligned")
            batch, history = rgbd_history.shape[:2]
            encoder_rgbd = rgbd_history
            if self.mask_goal_channel and rgbd_history.shape[2] > 2:
                encoder_rgbd = rgbd_history.clone()
                encoder_rgbd[:, :, 2] = 0
            image = self.image_encoder(
                encoder_rgbd.reshape(batch * history, *encoder_rgbd.shape[2:])
            )
            features.append(image.reshape(batch, history, -1))
        fused = torch.cat(features, dim=-1)
        if history_mask is not None:
            fused = fused * history_mask.unsqueeze(-1).to(fused.dtype)
        return self.head(_masked_gru_final(self.temporal, fused, history_mask))
