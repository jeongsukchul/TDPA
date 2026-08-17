from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEPLOYMENT_KEYS = frozenset({"rgbd", "proprio"})
PRIVILEGED_KEYS = frozenset(
    {
        "privileged",
        "mass",
        "friction",
        "physics",
        "object_state",
        "contact_state",
        "force_torque",
        "split",
        "task_id",
        "probe_id",
        "policy_id",
    }
)


@dataclass(frozen=True)
class Physics:
    mass: float
    friction: float

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.friction < 0:
            raise ValueError("friction must be non-negative")


class SyntheticManipulationEnv:
    """Deterministic surrogate used to validate experimental plumbing cheaply."""

    task = "base"

    def __init__(self, config: dict[str, Any], physics: Physics, seed: int = 0) -> None:
        self.config = config
        self.physics = physics
        self.dt = float(config.get("dt", 0.05))
        self.horizon = int(config.get("episode_length", 64))
        self.image_size = int(config.get("observation", {}).get("image_size", 16))
        self.force_limit = float(config.get("force_limit", 12.0))
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.t = 0
        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.ee_vel = np.zeros(3, dtype=np.float32)
        self.obj_pos = np.zeros(3, dtype=np.float32)
        self.obj_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.contact_force = np.zeros(3, dtype=np.float32)
        self.grip_width = np.float32(1.0)
        self.contact = False
        self.slipped = False
        self.dropped = False
        self._previous: dict[str, np.ndarray] = {}

    def reset(self) -> dict[str, np.ndarray]:
        self.t = 0
        self.ee_pos[:] = (-0.45, 0.0, 0.15)
        self.ee_vel[:] = 0
        self.obj_pos[:] = (0.0, 0.0, 0.04)
        self.obj_vel[:] = 0
        self.last_action[:] = 0
        self.contact_force[:] = 0
        self.grip_width = np.float32(1.0)
        self.contact = False
        self.slipped = False
        self.dropped = False
        self._snapshot()
        return self.observation()

    def _snapshot(self) -> None:
        self._previous = {
            "ee_vel": self.ee_vel.copy(),
            "obj_vel": self.obj_vel.copy(),
        }

    def observation(self) -> dict[str, np.ndarray]:
        proprio = np.concatenate(
            [self.ee_pos, self.ee_vel, [self.grip_width], self.last_action[:3]]
        ).astype(np.float32)
        return {"rgbd": self._render_rgbd(), "proprio": proprio}

    def privileged_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.obj_pos,
                self.obj_vel,
                self.contact_force,
                [float(self.contact)],
                self.ee_pos,
                self.last_action[:3],
            ]
        ).astype(np.float32)

    def response(self) -> np.ndarray:
        return np.concatenate(
            [
                self.ee_vel - self._previous["ee_vel"],
                self.last_action[:3] - self.ee_vel,
                self.obj_vel - self._previous["obj_vel"],
                self.contact_force,
            ]
        ).astype(np.float32)

    def _to_pixel(self, x: float, y: float) -> tuple[int, int]:
        size = self.image_size
        px = int(np.clip(round((x + 1.0) * 0.5 * (size - 1)), 0, size - 1))
        py = int(np.clip(round((y + 1.0) * 0.5 * (size - 1)), 0, size - 1))
        return py, px

    def target_position(self) -> np.ndarray:
        raise NotImplementedError

    def _render_rgbd(self) -> np.ndarray:
        # Visual appearance is deliberately independent of mass and friction.
        image = np.zeros((4, self.image_size, self.image_size), dtype=np.float32)
        obj_y, obj_x = self._to_pixel(float(self.obj_pos[0]), float(self.obj_pos[1]))
        ee_y, ee_x = self._to_pixel(float(self.ee_pos[0]), float(self.ee_pos[1]))
        target = self.target_position()
        target_y, target_x = self._to_pixel(float(target[0]), float(target[1]))
        image[0, obj_y, obj_x] = 1.0
        image[1, ee_y, ee_x] = 1.0
        image[2, target_y, target_x] = 1.0
        image[3, obj_y, obj_x] = np.clip(1.0 - self.obj_pos[2], 0.0, 1.0)
        return image

    def step(
        self, action: np.ndarray, controller: dict[str, float] | None = None
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._snapshot()
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action.shape}")
        self.last_action = action
        controller = controller or {}
        self._advance(action, controller)
        self.t += 1
        success = self.is_success()
        terminated = bool(success or self.dropped)
        truncated = self.t >= self.horizon
        reward = float(success) - 0.001 * float(np.linalg.norm(self.contact_force))
        info = self.metrics()
        return self.observation(), reward, terminated, truncated, info

    def _advance(self, action: np.ndarray, controller: dict[str, float]) -> None:
        raise NotImplementedError

    def is_success(self) -> bool:
        raise NotImplementedError

    def metrics(self) -> dict[str, float | bool]:
        raise NotImplementedError

