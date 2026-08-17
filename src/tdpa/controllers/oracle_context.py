from __future__ import annotations

from collections.abc import Mapping
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


class RobosuitePerfectContextOracle:
    """Fixed analytical controller schedule using only true mass and friction.

    The returned mapping is intentionally cell-agnostic. ``correction`` projects
    the analytical schedule into the configured bounds, and the evaluator routes
    it through :class:`AdapterActionMapper` as an independent execution bound.
    """

    def __init__(
        self,
        task: str,
        config: dict[str, Any],
        adapter_config: Mapping[str, Any],
        *,
        enabled: bool = False,
    ) -> None:
        if task not in {"push", "lift"}:
            raise ValueError(f"Unsupported task: {task}")
        if config.get("version") != "tdpa-robosuite-perfect-context-v1":
            raise ValueError("Unsupported robosuite oracle configuration")
        self.task = task
        self.config = config
        self.bounds = dict(adapter_config["bounds"])
        self.enabled = enabled

    @staticmethod
    def _positive_ratio(value: float, nominal: float, name: str) -> float:
        if not np.isfinite(value) or not np.isfinite(nominal) or value <= 0 or nominal <= 0:
            raise ValueError(f"Oracle {name} values must be finite and positive")
        return float(value / nominal)

    def raw_correction(self, physics: Physics) -> dict[str, Any]:
        if not self.enabled:
            raise PermissionError("Oracle physics access requires enabled=True")
        nominal_physics = self.config["nominal_physics"]
        mass_ratio = self._positive_ratio(
            float(physics.mass), float(nominal_physics["mass"]), "mass"
        )
        friction_ratio = self._positive_ratio(
            float(physics.friction), float(nominal_physics["friction"]), "friction"
        )
        schedule = self.config[self.task]
        if self.task == "push":
            velocity = friction_ratio ** float(schedule["velocity_friction_exponent"])
            stiffness = (
                100.0
                * mass_ratio ** float(schedule["stiffness_mass_exponent"])
                * friction_ratio ** float(schedule["stiffness_friction_exponent"])
            )
            return {
                "velocity_scale": velocity,
                "cartesian_residual": np.zeros(3, dtype=np.float32),
                "stiffness": stiffness,
                "damping": 15.0,
                "grip_force": 18.0,
            }

        velocity = mass_ratio ** float(
            schedule["velocity_mass_exponent"]
        ) * friction_ratio ** float(schedule["velocity_friction_exponent"])
        stiffness = 100.0 * mass_ratio ** float(schedule["stiffness_mass_exponent"])
        damping = 15.0 * mass_ratio ** float(schedule["damping_mass_exponent"])
        grip_force = (
            18.0
            * mass_ratio ** float(schedule["grip_mass_exponent"])
            * friction_ratio ** float(schedule["grip_friction_exponent"])
        )
        return {
            "velocity_scale": velocity,
            "cartesian_residual": np.zeros(3, dtype=np.float32),
            "stiffness": stiffness,
            "damping": damping,
            "grip_force": grip_force,
        }

    def correction(self, physics: Physics) -> dict[str, Any]:
        raw = self.raw_correction(physics)
        bounded = dict(raw)
        velocity_low, velocity_high = map(float, self.bounds["velocity_scale"])
        bounded["velocity_scale"] = float(
            np.clip(raw["velocity_scale"], velocity_low, velocity_high)
        )
        residual_limit = float(self.bounds["residual_max"])
        bounded["cartesian_residual"] = np.clip(
            np.asarray(raw["cartesian_residual"], dtype=np.float32),
            -residual_limit,
            residual_limit,
        )
        for key in ("stiffness", "damping", "grip_force"):
            low, high = map(float, self.bounds[key])
            bounded[key] = float(np.clip(raw[key], low, high))
        return bounded
