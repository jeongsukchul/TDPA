from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_TASK_CODES = {"push": 0x50555348, "lift": 0x4C494654}


@dataclass(frozen=True)
class ResetState:
    """Privileged, index-addressable simulator reset state used only for auditing."""

    task: str
    seed: int
    episode_index: int
    robot_qpos: tuple[float, ...]
    object_position: tuple[float, float, float]
    object_quaternion: tuple[float, float, float, float]
    target_position: tuple[float, float, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_reset_state(
    task: str,
    *,
    seed: int,
    episode_index: int,
    robot_qpos: np.ndarray,
    table_height: float,
    object_half_height: float,
    config: dict[str, Any],
) -> ResetState:
    """Resolve a reset without global RNG or physics-dependent call order."""
    if task not in _TASK_CODES:
        raise ValueError(f"Unknown task: {task}")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    sequence = np.random.SeedSequence(
        [_TASK_CODES[task], int(seed) & 0xFFFFFFFF, int(episode_index) & 0xFFFFFFFF]
    )
    rng = np.random.default_rng(sequence)
    reset = config.get("reset", {})
    center = np.asarray(reset.get("object_xy_center", [0.0, 0.0]), dtype=np.float64)
    half_range = np.asarray(reset.get("object_xy_half_range", [0.025, 0.025]), dtype=np.float64)
    if center.shape != (2,) or half_range.shape != (2,) or np.any(half_range < 0):
        raise ValueError("robosuite reset object bounds must be two non-negative half-ranges")
    object_xy = center + rng.uniform(-half_range, half_range)
    nominal_robot_qpos = np.asarray(robot_qpos, dtype=np.float64).reshape(-1)
    joint_half_range = np.asarray(reset.get("robot_qpos_half_range", 0.0), dtype=np.float64)
    if joint_half_range.ndim == 0:
        joint_half_range = np.full(nominal_robot_qpos.shape, float(joint_half_range))
    if joint_half_range.shape != nominal_robot_qpos.shape or np.any(joint_half_range < 0):
        raise ValueError("robot_qpos_half_range must be non-negative and scalar or joint-sized")
    resolved_robot_qpos = nominal_robot_qpos + rng.uniform(-joint_half_range, joint_half_range)
    yaw_range = float(reset.get("object_yaw_range", np.pi))
    yaw = float(rng.uniform(-yaw_range, yaw_range))
    quat = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))

    target = np.asarray(config["target_position"], dtype=np.float64).copy()
    target_jitter = np.asarray(reset.get("target_xyz_half_range", [0.0, 0.0, 0.0]))
    if target.shape != (3,) or target_jitter.shape != (3,) or np.any(target_jitter < 0):
        raise ValueError("robosuite target position and jitter must be three-vectors")
    target += rng.uniform(-target_jitter, target_jitter)

    return ResetState(
        task=task,
        seed=int(seed),
        episode_index=int(episode_index),
        robot_qpos=tuple(float(value) for value in resolved_robot_qpos),
        object_position=(
            float(object_xy[0]),
            float(object_xy[1]),
            float(table_height + object_half_height),
        ),
        object_quaternion=quat,
        target_position=tuple(float(value) for value in target),
    )
