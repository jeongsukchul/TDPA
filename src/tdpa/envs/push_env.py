from __future__ import annotations

from typing import Any

import numpy as np

from tdpa.envs.base import Physics, SyntheticManipulationEnv
from tdpa.utils.geometry import clip_norm


class PushEnv(SyntheticManipulationEnv):
    task = "push"

    def __init__(self, config: dict[str, Any], physics: Physics, seed: int = 0) -> None:
        super().__init__(config, physics, seed)
        self.goal = np.asarray(config.get("target_position", [0.75, 0.0]), dtype=np.float32)
        self.tolerance = float(config.get("success_tolerance", 0.08))

    def target_position(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1], 0.04], dtype=np.float32)

    def _advance(self, action: np.ndarray, controller: dict[str, float]) -> None:
        velocity_scale = float(np.clip(controller.get("velocity_scale", 1.0), 0.1, 2.0))
        stiffness = float(np.clip(controller.get("stiffness", 100.0), 10.0, 300.0))
        desired_velocity = clip_norm(action[:3] * 0.8 * velocity_scale, 1.2)
        self.ee_vel += 0.55 * (desired_velocity - self.ee_vel)
        self.ee_pos += self.ee_vel * self.dt
        relative = self.ee_pos[:2] - self.obj_pos[:2]
        self.contact = bool(
            abs(relative[1]) < 0.13
            and -0.15 < relative[0] < 0.07
            and self.ee_pos[2] < 0.22
            and self.ee_vel[0] > -0.05
        )
        if self.contact:
            direction = self.ee_vel[:2]
            drive = (stiffness / 100.0) * 13.0 * direction
            friction_drag = 9.81 * self.physics.friction * np.tanh(self.obj_vel[:2] * 10.0)
            acceleration = (drive / self.physics.mass) - friction_drag
            self.obj_vel[:2] += acceleration.astype(np.float32) * self.dt
            self.contact_force[:2] = drive.astype(np.float32)
            # A unilateral constraint prevents pusher tunnelling while leaving
            # object motion sensitive to mass and friction.
            self.ee_pos[0] = min(self.ee_pos[0], self.obj_pos[0] - 0.055)
        else:
            self.contact_force[:] = 0
            speed = np.linalg.norm(self.obj_vel[:2])
            if speed > 0:
                decel = min(speed, 9.81 * self.physics.friction * self.dt)
                self.obj_vel[:2] *= max(0.0, 1.0 - decel / speed)
        self.obj_vel[:2] *= 0.985
        self.obj_pos[:2] += self.obj_vel[:2] * self.dt

    def is_success(self) -> bool:
        return bool(np.linalg.norm(self.obj_pos[:2] - self.goal) <= self.tolerance)

    def metrics(self) -> dict[str, float | bool]:
        error = float(np.linalg.norm(self.obj_pos[:2] - self.goal))
        overshoot = float(max(0.0, self.obj_pos[0] - self.goal[0]))
        force = float(np.linalg.norm(self.contact_force))
        return {
            "success": self.is_success(),
            "final_error": error,
            "completion_time": self.t * self.dt,
            "contact_force": force,
            "overshoot": overshoot,
            "force_violation": force > self.force_limit,
        }
