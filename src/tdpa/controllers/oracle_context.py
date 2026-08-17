from __future__ import annotations

from typing import Any

import numpy as np

from tdpa.envs.base import Physics


class OracleContextAdapter:
    """Transparent perfect-context upper bound; never a deployment component."""

    def __init__(self, task: str, enabled: bool = False) -> None:
        self.task = task
        self.enabled = enabled

    def correction(self, physics: Physics) -> dict[str, Any]:
        if not self.enabled:
            raise PermissionError("Oracle physics access requires enabled=True")
        if self.task == "push":
            scale = min(1.0, np.sqrt(max(physics.friction, 0.05) / 0.55))
            return {
                "velocity_scale": float(np.clip(scale, 0.35, 1.5)),
                "cartesian_residual": np.zeros(3, dtype=np.float32),
                "stiffness": float(
                    np.clip(100.0 * physics.mass * physics.friction / 0.55, 40.0, 180.0)
                ),
            }
        if self.task == "lift":
            required_grip = 1.3 * physics.mass * 9.81 / (1.8 * max(physics.friction, 0.05))
            return {
                "velocity_scale": float(np.clip(min(1.0, 1.0 / np.sqrt(physics.mass)), 0.3, 1.35)),
                "cartesian_residual": np.zeros(3, dtype=np.float32),
                "stiffness": float(np.clip(110.0 * physics.mass, 50.0, 220.0)),
                "damping": float(np.clip(12.0 + 8.0 * physics.mass, 5.0, 35.0)),
                "grip_force": float(np.clip(required_grip, 3.0, 50.0)),
            }
        raise ValueError(f"Unsupported task: {self.task}")
