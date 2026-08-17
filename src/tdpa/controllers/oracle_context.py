from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tdpa.envs.base import Physics
from tdpa.policies.frozen_nominal import assert_deployment_observation


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


@dataclass(frozen=True)
class OracleV2Decision:
    """Auditable raw and bounded command for one causal control step."""

    raw_correction: dict[str, Any]
    correction: dict[str, Any]
    phase: str


class RobosuitePerfectContextOracleV2:
    """Phase-preserving upper bound with mass/friction as its only privilege.

    RGB-D is validated but never inspected. Proprioception and the frozen nominal
    action are deployment inputs available to a future learned adapter. Physics is
    the sole privileged input, and no split or OOD-cell label is accepted.
    """

    def __init__(
        self,
        task: str,
        config: Mapping[str, Any],
        adapter_config: Mapping[str, Any],
        *,
        enabled: bool = False,
    ) -> None:
        if task not in {"push", "lift"}:
            raise ValueError(f"Unsupported task: {task}")
        if config.get("version") != "tdpa-robosuite-perfect-context-v2":
            raise ValueError("Unsupported robosuite oracle-v2 configuration")
        self.task = task
        self.config = dict(config)
        self.bounds = dict(adapter_config["bounds"])
        self.enabled = enabled
        self._lift_phase_active = False
        self._lift_close_steps = 0

    def reset(self) -> None:
        self._lift_phase_active = False
        self._lift_close_steps = 0

    @staticmethod
    def _positive_ratio(value: float, nominal: float, name: str) -> float:
        if not np.isfinite(value) or not np.isfinite(nominal) or value <= 0 or nominal <= 0:
            raise ValueError(f"Oracle-v2 {name} values must be finite and positive")
        return float(value / nominal)

    def _bounded(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        bounded = dict(raw)
        velocity_low, velocity_high = map(float, self.bounds["velocity_scale"])
        bounded["velocity_scale"] = float(
            np.clip(float(raw["velocity_scale"]), velocity_low, velocity_high)
        )
        residual_limit = float(self.bounds["residual_max"])
        bounded["cartesian_residual"] = np.clip(
            np.asarray(raw["cartesian_residual"], dtype=np.float32),
            -residual_limit,
            residual_limit,
        )
        for key in ("stiffness", "damping", "grip_force"):
            low, high = map(float, self.bounds[key])
            bounded[key] = float(np.clip(float(raw[key]), low, high))
        return bounded

    def decide(
        self,
        physics: Physics,
        nominal_action: np.ndarray,
        observation: Mapping[str, np.ndarray],
    ) -> OracleV2Decision:
        if not self.enabled:
            raise PermissionError("Oracle physics access requires enabled=True")
        assert_deployment_observation(observation)
        action = np.asarray(nominal_action, dtype=np.float32)
        proprio = np.asarray(observation["proprio"], dtype=np.float32)
        if action.shape != (4,) or proprio.shape != (10,):
            raise ValueError("Oracle-v2 expects a 4-D action and 10-D proprioception")
        if not np.isfinite(action).all() or not np.isfinite(proprio).all():
            raise ValueError("Oracle-v2 deployment inputs must be finite")

        nominal = self.config["nominal_physics"]
        mass_ratio = self._positive_ratio(float(physics.mass), float(nominal["mass"]), "mass")
        friction_ratio = self._positive_ratio(
            float(physics.friction), float(nominal["friction"]), "friction"
        )
        if self.task == "push":
            schedule = self.config["push"]
            friction_velocity = friction_ratio ** float(
                schedule["velocity_friction_exponent"]
            )
            velocity = 1.0 + float(schedule["velocity_correction_gain"]) * (
                friction_velocity - 1.0
            )
            requested_velocity = float(
                np.clip(
                    velocity,
                    float(schedule["velocity_scale_minimum"]),
                    float(schedule["velocity_scale_maximum"]),
                )
            )
            velocity = requested_velocity
            peak_action = float(np.max(np.abs(action[:3])))
            if peak_action > 0.0:
                velocity = min(
                    velocity,
                    float(schedule["action_headroom_margin"]) / peak_action,
                )
            stiffness = (
                100.0
                * mass_ratio ** float(schedule["stiffness_mass_exponent"])
                * friction_ratio ** float(schedule["stiffness_friction_exponent"])
            )
            raw = {
                "velocity_scale": requested_velocity,
                "cartesian_residual": np.zeros(3, dtype=np.float32),
                "stiffness": stiffness,
                "damping": 15.0,
                "grip_force": 18.0,
            }
            bounded = self._bounded(raw)
            bounded["velocity_scale"] = velocity
            return OracleV2Decision(raw, bounded, "push")

        schedule = self.config["lift"]
        close_requested = float(action[3]) >= float(schedule["close_command_threshold"])
        gripper_narrow = float(proprio[6]) <= float(schedule["gripper_width_threshold"])
        upward_requested = float(action[2]) > float(schedule["upward_action_threshold"])
        if close_requested:
            self._lift_phase_active = True
            self._lift_close_steps += 1
        mass_excess = max(mass_ratio - 1.0, 0.0)
        friction_deficit = max(friction_ratio**-0.5 - 1.0, 0.0)
        settle_steps = int(
            np.ceil(
                float(schedule["settle_steps_base"])
                + float(schedule["settle_steps_mass_gain"]) * mass_excess
                + float(schedule["settle_steps_friction_gain"]) * friction_deficit
            )
        )
        if not gripper_narrow:
            settle_steps += int(schedule["settle_wide_gripper_extra_steps"])
        settle_phase = self._lift_phase_active and self._lift_close_steps <= settle_steps
        lift_phase = self._lift_phase_active and not settle_phase and upward_requested
        residual = np.zeros(3, dtype=np.float32)
        if settle_phase and upward_requested:
            residual[2] = -min(
                max(float(action[2]), 0.0),
                float(schedule["settle_vertical_cancel_maximum"]),
            )
        elif lift_phase:
            requested = (
                float(schedule["vertical_residual_gain"])
                * mass_excess
                * max(float(action[2]), 0.0)
            )
            residual[2] = float(
                np.clip(requested, 0.0, float(schedule["vertical_residual_maximum"]))
            )
        raw = {
            "velocity_scale": float(schedule["velocity_scale"]),
            "cartesian_residual": residual,
            "stiffness": 100.0
            * mass_ratio ** float(schedule["stiffness_mass_exponent"]),
            "damping": 15.0 * mass_ratio ** float(schedule["damping_mass_exponent"]),
            "grip_force": (
                18.0
                * mass_ratio ** float(schedule["grip_mass_exponent"])
                * friction_ratio ** float(schedule["grip_friction_exponent"])
            ),
        }
        bounded = self._bounded(raw)
        margin = float(schedule["action_headroom_margin"])
        motion = action[:3] * float(bounded["velocity_scale"])
        bounded["cartesian_residual"] = np.clip(
            np.asarray(bounded["cartesian_residual"]),
            -margin - motion,
            margin - motion,
        ).astype(np.float32)
        if settle_phase:
            phase = "settle"
        elif lift_phase:
            phase = "lift"
        else:
            phase = "closed" if self._lift_phase_active else "approach"
        return OracleV2Decision(raw, bounded, phase)
