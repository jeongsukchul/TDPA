from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tdpa.models.history_encoder import HistoryEncoder


class MultiTaskRMA(nn.Module):
    """Capacity-explicit two-task RMA baseline with task conditioning."""

    def __init__(
        self,
        encoder_config: Mapping[str, Any],
        output_dim: int = 7,
        task_count: int = 2,
        task_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.context = HistoryEncoder(
            proprio_dim=int(encoder_config["proprio_dim"]),
            action_dim=int(encoder_config["action_dim"]),
            latent_dim=int(encoder_config["latent_dim"]),
            hidden_dim=int(encoder_config["hidden_dim"]),
            image_channels=int(encoder_config.get("image_channels", 4)),
            use_rgbd=bool(encoder_config.get("use_rgbd", True)),
            mask_goal_channel=bool(encoder_config.get("mask_goal_channel", True)),
        )
        latent = int(encoder_config["latent_dim"])
        hidden = int(encoder_config["hidden_dim"])
        self.task_embedding = nn.Embedding(task_count, task_embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(latent + task_embedding_dim + 4 + 10, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(
        self,
        rgbd_history: torch.Tensor | None,
        proprio_history: torch.Tensor,
        action_history: torch.Tensor,
        nominal_action: torch.Tensor,
        proprio_state: torch.Tensor,
        task_id: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.context(rgbd_history, proprio_history, action_history, history_mask)
        task = self.task_embedding(task_id)
        return self.head(torch.cat([context, task, nominal_action, proprio_state], dim=-1))
