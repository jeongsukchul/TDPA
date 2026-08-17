from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AppliedAction:
    action: np.ndarray
    controller: dict[str, float]
    saturated: bool

    @property
    def execution_command(self) -> np.ndarray:
        """Canonical post-clipping command consumed by response models."""
        return np.concatenate(
            [
                self.action,
                [
                    self.controller["stiffness"] / 300.0,
                    self.controller["damping"] / 60.0,
                    self.controller["grip_force"] / 60.0,
                ],
            ]
        ).astype(np.float32)


def _finite_scalar(value: object, default: float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    array = np.asarray(value)
    scalar = float(array.reshape(-1)[0])
    return scalar if np.isfinite(scalar) else default


def _finite_vector(value: object, size: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size < size:
        vector = np.pad(vector, (0, size - vector.size))
    return np.nan_to_num(vector[:size], nan=0.0, posinf=0.0, neginf=0.0)


def _contains_nonfinite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        return not bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return True


class AdapterActionMapper:
    """Second, controller-side bound on learned physical corrections."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.bounds = dict(config["bounds"])  # type: ignore[arg-type]
        self.outputs = dict(config["outputs"])  # type: ignore[arg-type]
        self.nominal = dict(config.get("nominal", {}))  # type: ignore[arg-type]

    def apply(
        self, nominal_action: np.ndarray, correction: Mapping[str, object] | None = None
    ) -> AppliedAction:
        input_nonfinite = _contains_nonfinite(nominal_action)
        nominal = np.nan_to_num(
            np.asarray(nominal_action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0
        )
        if nominal.shape != (4,):
            raise ValueError(f"Expected nominal action (4,), got {nominal.shape}")
        correction = correction or {}
        input_nonfinite = input_nonfinite or any(
            _contains_nonfinite(value) for value in correction.values()
        )
        nominal_v = float(self.nominal.get("velocity_scale", 1.0))
        requested_v = _finite_scalar(correction.get("velocity_scale", nominal_v), nominal_v)
        v_low, v_high = map(float, self.bounds["velocity_scale"])
        velocity_scale = float(np.clip(requested_v, v_low, v_high))
        residual_limit = float(self.bounds["residual_max"])
        requested_residual = _finite_vector(correction.get("cartesian_residual", 0.0), 3)
        residual = np.clip(requested_residual, -residual_limit, residual_limit)
        motion = nominal[:3] * velocity_scale + residual
        action = np.concatenate([np.clip(motion, -1.0, 1.0), [np.clip(nominal[3], -1.0, 1.0)]])
        controller: dict[str, float] = {"velocity_scale": 1.0}
        controller_saturated = False
        for key in ("stiffness", "damping", "grip_force"):
            low, high = map(float, self.bounds[key])
            default = float(self.nominal.get(key, (low + high) / 2.0))
            requested = correction.get(key, default) if self.outputs.get(key, False) else default
            requested_value = _finite_scalar(requested, default)
            controller[key] = float(np.clip(requested_value, low, high))
            controller_saturated = controller_saturated or requested_value != controller[key]
        saturated = bool(
            input_nonfinite
            or requested_v != velocity_scale
            or np.any(requested_residual != residual)
            or np.any(np.abs(nominal[:3] * velocity_scale + residual) > 1.0)
            or abs(float(nominal[3])) > 1.0
            or controller_saturated
        )
        return AppliedAction(action.astype(np.float32), controller, saturated)
