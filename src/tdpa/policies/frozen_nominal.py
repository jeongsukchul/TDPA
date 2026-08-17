from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tdpa.envs.base import DEPLOYMENT_KEYS, PRIVILEGED_KEYS
from tdpa.utils.geometry import clip_norm


def assert_deployment_observation(observation: Mapping[str, np.ndarray]) -> None:
    keys = set(observation)
    forbidden = keys & PRIVILEGED_KEYS
    if forbidden:
        raise RuntimeError(f"Privileged deployment fields rejected: {sorted(forbidden)}")
    unknown = keys - DEPLOYMENT_KEYS
    if unknown:
        raise RuntimeError(f"Unknown deployment fields rejected: {sorted(unknown)}")
    missing = DEPLOYMENT_KEYS - keys
    if missing:
        raise RuntimeError(f"Missing deployment fields: {sorted(missing)}")


def _pixel_to_world(pixel: tuple[int, int], size: int) -> np.ndarray:
    y, x = pixel
    return np.array([2.0 * x / (size - 1) - 1.0, 2.0 * y / (size - 1) - 1.0])


def _locate(channel: np.ndarray) -> np.ndarray:
    positions = np.argwhere(channel == channel.max())
    if channel.max() <= 0 or len(positions) == 0:
        raise RuntimeError("Synthetic vision marker is not visible")
    return _pixel_to_world(tuple(positions[0]), channel.shape[-1])


class FrozenNominalPolicy:
    """Physics-blind visual servo used as deterministic benchmark scaffolding."""

    def __init__(self, task: str, task_config: Mapping[str, object]) -> None:
        if task not in {"push", "lift"}:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task
        self.config = task_config
        # A scalar assertion is easier to audit than relying on optimizer conventions.
        self.frozen = True

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        assert_deployment_observation(observation)
        if not self.frozen:
            raise RuntimeError("Base policy must remain frozen during physical adaptation")
        image = np.asarray(observation["rgbd"], dtype=np.float32)
        proprio = np.asarray(observation["proprio"], dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != 4 or proprio.shape != (10,):
            raise ValueError("Unexpected deployment observation shape")
        obj_xy = _locate(image[0])
        target_xy = _locate(image[2])
        ee = proprio[:3]
        if self.task == "push":
            # Move behind the object first, then drive the object through the target.
            behind = np.array([obj_xy[0] - 0.08, obj_xy[1], 0.10], dtype=np.float32)
            close_xy = np.linalg.norm(ee[:2] - behind[:2]) < 0.11
            waypoint = (
                np.array([target_xy[0] + 0.08, target_xy[1], 0.10], dtype=np.float32)
                if close_xy or ee[0] >= obj_xy[0] - 0.10
                else behind
            )
            motion = clip_norm((waypoint - ee) * 4.5, 1.0)
            return np.array([*motion, -1.0], dtype=np.float32)

        obj_depth = float(image[3][image[0].argmax() // image.shape[-1], image[0].argmax() % image.shape[-1]])
        obj_z = 1.0 - obj_depth
        obj = np.array([obj_xy[0], obj_xy[1], obj_z], dtype=np.float32)
        goal = np.array(
            [
                float(self.config.get("target_x", 0.45)),
                0.0,
                float(self.config.get("target_height", 0.65)),
            ],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(ee - obj))
        if distance > 0.11 and obj_z < 0.12:
            waypoint = obj + np.array([0.0, 0.0, 0.03], dtype=np.float32)
            grip = -1.0
        else:
            waypoint = goal
            grip = 1.0
        motion = clip_norm((waypoint - ee) * 3.8, 1.0)
        return np.array([*motion, grip], dtype=np.float32)

