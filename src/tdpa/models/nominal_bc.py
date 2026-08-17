"""Task-specific visual action-chunk behavior cloning model."""

from __future__ import annotations

from itertools import pairwise
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class ActionChunkPolicy(Protocol):
    """Small interface also implementable by a future diffusion policy."""

    def predict_action_chunk(
        self,
        rgbd_history: torch.Tensor,
        proprio_history: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> torch.Tensor: ...


class VisualActionChunkBC(nn.Module):
    """Encode two causal observations and predict a bounded Cartesian action chunk."""

    family = "visual_action_chunk_bc"

    def __init__(
        self,
        *,
        history_length: int = 2,
        action_horizon: int = 8,
        proprio_dim: int = 10,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        if min(history_length, action_horizon, proprio_dim, action_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        self.history_length = int(history_length)
        self.action_horizon = int(action_horizon)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        channels = (4, 32, 64, 128, 256)
        blocks: list[nn.Module] = []
        for source, target in pairwise(channels):
            blocks.extend(
                [
                    nn.Conv2d(source, target, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(min(8, target), target),
                    nn.SiLU(),
                ]
            )
        self.vision = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proprio = nn.Sequential(
            nn.Linear(self.proprio_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.temporal = nn.GRU(256 + 64, 256, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self.action_horizon * self.action_dim),
            nn.Tanh(),
        )

    def model_config(self) -> dict[str, int]:
        return {
            "history_length": self.history_length,
            "action_horizon": self.action_horizon,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
        }

    def forward(
        self,
        rgbd_history: torch.Tensor,
        proprio_history: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        if rgbd_history.ndim != 5 or rgbd_history.shape[2] != 4:
            raise ValueError("rgbd_history must have shape [batch, history, 4, height, width]")
        batch, history = rgbd_history.shape[:2]
        if history != self.history_length:
            raise ValueError(f"Expected history length {self.history_length}, got {history}")
        if proprio_history.shape != (batch, history, self.proprio_dim):
            raise ValueError("proprio_history shape does not match model configuration")
        if observation_mask.shape != (batch, history):
            raise ValueError("observation_mask shape does not match history")
        if not torch.all(observation_mask[:, -1]):
            raise ValueError("The most recent observation must always be valid")
        images = rgbd_history.reshape(batch * history, *rgbd_history.shape[2:])
        vision = self.vision(images).reshape(batch, history, -1)
        state = self.proprio(proprio_history)
        mask = observation_mask.to(dtype=vision.dtype).unsqueeze(-1)
        sequence = torch.cat([vision, state], dim=-1) * mask
        encoded, _ = self.temporal(sequence)
        actions = self.action_head(encoded[:, -1])
        return actions.reshape(batch, self.action_horizon, self.action_dim)

    def predict_action_chunk(
        self,
        rgbd_history: torch.Tensor,
        proprio_history: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self(rgbd_history, proprio_history, observation_mask)
