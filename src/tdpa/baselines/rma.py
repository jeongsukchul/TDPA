from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from tdpa.models.history_encoder import HistoryEncoder
from tdpa.models.physical_adapter import PhysicalAdapter


class PerTaskRMA(nn.Module):
    """Task-local history encoder and adapter trained jointly from scratch."""

    def __init__(self, encoder_config: Mapping[str, Any], adapter_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoder = HistoryEncoder(
            proprio_dim=int(encoder_config["proprio_dim"]),
            action_dim=int(encoder_config["action_dim"]),
            latent_dim=int(encoder_config["latent_dim"]),
            hidden_dim=int(encoder_config["hidden_dim"]),
            image_channels=int(encoder_config.get("image_channels", 4)),
            use_rgbd=bool(encoder_config.get("use_rgbd", True)),
            mask_goal_channel=bool(encoder_config.get("mask_goal_channel", True)),
        )
        self.adapter = PhysicalAdapter(adapter_config)
