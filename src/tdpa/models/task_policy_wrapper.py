from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from tdpa.models.history_encoder import HistoryEncoder
from tdpa.models.physical_adapter import PhysicalAdapter
from tdpa.policies.frozen_nominal import FrozenNominalPolicy, assert_deployment_observation


class TaskPolicyWrapper:
    """Compose a frozen task policy, frozen encoder, and trainable small adapter."""

    def __init__(
        self,
        base_policy: FrozenNominalPolicy,
        encoder: HistoryEncoder,
        adapter: PhysicalAdapter,
    ) -> None:
        self.base_policy = base_policy
        self.encoder = encoder.eval()
        self.adapter = adapter
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        observation: Mapping[str, np.ndarray],
        rgbd_history: torch.Tensor | None,
        proprio_history: torch.Tensor,
        action_history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        assert_deployment_observation(observation)
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            raise RuntimeError("Shared encoder must remain frozen during adapter training")
        nominal = self.base_policy(observation)
        with torch.no_grad():
            latent = self.encoder(rgbd_history, proprio_history, action_history, history_mask)
        nominal_tensor = torch.as_tensor(nominal, dtype=proprio_history.dtype, device=proprio_history.device).view(1, -1)
        current_proprio = torch.as_tensor(
            observation["proprio"], dtype=proprio_history.dtype, device=proprio_history.device
        ).view(1, -1)
        return nominal, self.adapter(nominal_tensor, latent, current_proprio)

