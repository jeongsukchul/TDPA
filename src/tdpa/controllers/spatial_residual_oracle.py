"""Privileged spatial residual used only to audit the current adapter interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tdpa.controllers.oracle_context import RobosuitePerfectContextOracleV2
from tdpa.envs.base import Physics
from tdpa.policies.frozen_nominal import assert_deployment_observation
from tdpa.policies.privileged_expert import PrivilegedScriptedExpert


@dataclass(frozen=True)
class SpatialResidualDecision:
    correction: dict[str, Any]
    desired_residual: np.ndarray
    residual_projected: bool
    residual_near_bound: bool
    expert_action: np.ndarray
    expert_phase: str
    physics_phase: str
    gripper_disagreement: bool


def bounded_spatial_residual(
    nominal_motion: np.ndarray,
    expert_motion: np.ndarray,
    *,
    velocity_scale: float,
    residual_limit: float,
    near_bound_fraction: float,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    nominal = np.asarray(nominal_motion, dtype=np.float32)
    expert = np.asarray(expert_motion, dtype=np.float32)
    if nominal.shape != (3,) or expert.shape != (3,):
        raise ValueError("Spatial residual expects two 3-D motions")
    if not np.isfinite(nominal).all() or not np.isfinite(expert).all():
        raise ValueError("Spatial residual motions must be finite")
    if not np.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ValueError("Spatial residual velocity scale must be finite and positive")
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("Spatial residual limit must be finite and positive")
    if not 0.0 < near_bound_fraction <= 1.0:
        raise ValueError("near_bound_fraction must lie in (0, 1]")
    desired = expert - nominal * float(velocity_scale)
    bounded = np.clip(desired, -residual_limit, residual_limit).astype(np.float32)
    projected = bool(np.any(np.abs(desired) > residual_limit))
    near_bound = bool(np.any(np.abs(bounded) >= residual_limit * near_bound_fraction))
    return desired.astype(np.float32), bounded, projected, near_bound


class PrivilegedSpatialResidualOracle:
    """Bounded residual oracle with explicit privileged spatial-state access.

    The nominal policy's gripper command is never replaced. Object pose, target
    pose, and grasp state enter only through ``PrivilegedScriptedExpert`` and
    are recorded as oracle-only inputs.
    """

    version = "tdpa-privileged-spatial-residual-oracle-v1"

    def __init__(
        self,
        config: Mapping[str, Any],
        physics_config: Mapping[str, Any],
        adapter_config: Mapping[str, Any],
        *,
        position_delta_limit: float,
        enabled: bool = False,
    ) -> None:
        if config.get("version") != "tdpa-robosuite-spatial-residual-v1":
            raise ValueError("Unsupported privileged spatial-residual configuration")
        if config.get("task") != "lift":
            raise ValueError("Spatial-residual oracle is locked to Lift")
        correction = config["correction"]
        residual_limit = float(adapter_config["bounds"]["residual_max"])
        if float(correction["residual_max"]) != residual_limit:
            raise ValueError("Spatial residual limit does not match the adapter contract")
        if not bool(correction["preserve_nominal_gripper"]):
            raise ValueError("Spatial oracle must preserve the nominal gripper command")
        if not bool(correction["reuse_physics_oracle_controller_schedule"]):
            raise ValueError("Spatial oracle must reuse the physics-oracle controller schedule")
        self.config = dict(config)
        self.residual_limit = residual_limit
        self.near_bound_fraction = float(correction["residual_near_bound_fraction"])
        self.enabled = enabled
        self.expert = PrivilegedScriptedExpert("lift", position_delta_limit=position_delta_limit)
        self.physics_oracle = RobosuitePerfectContextOracleV2(
            "lift", physics_config, adapter_config, enabled=enabled
        )

    def reset(self) -> None:
        self.expert.reset()
        self.physics_oracle.reset()

    def decide(
        self,
        env: Any,
        physics: Physics,
        nominal_action: np.ndarray,
        observation: Mapping[str, np.ndarray],
    ) -> SpatialResidualDecision:
        if not self.enabled:
            raise PermissionError("Privileged spatial access requires enabled=True")
        assert_deployment_observation(observation)
        nominal = np.asarray(nominal_action, dtype=np.float32)
        if nominal.shape != (4,) or not np.isfinite(nominal).all():
            raise ValueError("Spatial oracle expects a finite 4-D nominal action")
        physics_decision = self.physics_oracle.decide(physics, nominal, observation)
        expert_decision = self.expert.act(env, dict(observation))
        desired, residual, projected, near_bound = bounded_spatial_residual(
            nominal[:3],
            expert_decision.action[:3],
            velocity_scale=float(physics_decision.correction["velocity_scale"]),
            residual_limit=self.residual_limit,
            near_bound_fraction=self.near_bound_fraction,
        )
        correction = dict(physics_decision.correction)
        correction["cartesian_residual"] = residual
        return SpatialResidualDecision(
            correction=correction,
            desired_residual=desired,
            residual_projected=projected,
            residual_near_bound=near_bound,
            expert_action=expert_decision.action.copy(),
            expert_phase=expert_decision.phase,
            physics_phase=physics_decision.phase,
            gripper_disagreement=not np.isclose(
                float(expert_decision.action[3]), float(nominal[3]), rtol=0.0, atol=1e-6
            ),
        )
