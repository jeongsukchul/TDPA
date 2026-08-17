from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class PhysicalAdapter(nn.Module):
    """Small task-specific adapter with bounded physical outputs."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.outputs = dict(config["outputs"])
        self.bounds = dict(config["bounds"])
        hidden = int(config.get("hidden_dim", 64))
        input_dim = int(config["nominal_action_dim"]) + int(config["latent_dim"]) + int(config["proprio_dim"])
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.heads = nn.ModuleDict(
            {
                "cartesian_residual": nn.Linear(hidden, 3),
                "velocity_scale": nn.Linear(hidden, 1),
                "stiffness": nn.Linear(hidden, 1),
                "damping": nn.Linear(hidden, 1),
                "grip_force": nn.Linear(hidden, 1),
            }
        )

    @staticmethod
    def _bounded(raw: torch.Tensor, bounds: list[float] | tuple[float, float]) -> torch.Tensor:
        low, high = map(float, bounds)
        return low + torch.sigmoid(raw) * (high - low)

    def forward(
        self,
        nominal_action: torch.Tensor,
        dynamics_latent: torch.Tensor,
        proprio_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return bounded correction parameters using no privileged context."""
        if nominal_action.ndim != 2 or dynamics_latent.ndim != 2 or proprio_state.ndim != 2:
            raise ValueError("Adapter inputs must all have shape [B,D]")
        hidden = self.backbone(torch.cat([nominal_action, dynamics_latent, proprio_state], dim=-1))
        residual_limit = float(self.bounds["residual_max"])
        values = {
            "cartesian_residual": residual_limit * torch.tanh(self.heads["cartesian_residual"](hidden)),
            "velocity_scale": self._bounded(self.heads["velocity_scale"](hidden), self.bounds["velocity_scale"]),
            "stiffness": self._bounded(self.heads["stiffness"](hidden), self.bounds["stiffness"]),
            "damping": self._bounded(self.heads["damping"](hidden), self.bounds["damping"]),
            "grip_force": self._bounded(self.heads["grip_force"](hidden), self.bounds["grip_force"]),
        }
        # Disabled outputs remain fixed at neutral values so callers can depend
        # on a stable dictionary interface without training unused heads.
        neutral = {
            "cartesian_residual": 0.0,
            "velocity_scale": 1.0,
            "stiffness": sum(self.bounds["stiffness"]) / 2.0,
            "damping": sum(self.bounds["damping"]) / 2.0,
            "grip_force": sum(self.bounds["grip_force"]) / 2.0,
        }
        for name, enabled in self.outputs.items():
            if not enabled:
                values[name] = torch.full_like(values[name], float(neutral[name]))
        return values

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

