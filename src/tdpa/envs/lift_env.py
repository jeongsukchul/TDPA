from __future__ import annotations

from typing import Any

import numpy as np

from tdpa.envs.base import Physics, SyntheticManipulationEnv
from tdpa.utils.geometry import clip_norm


class LiftEnv(SyntheticManipulationEnv):
    task = "lift"

    def __init__(self, config: dict[str, Any], physics: Physics, seed: int = 0) -> None:
        super().__init__(config, physics, seed)
        self.goal = np.array(
            [float(config.get("target_x", 0.45)), 0.0, float(config.get("target_height", 0.65))],
            dtype=np.float32,
        )
        self.tolerance = float(config.get("success_tolerance", 0.09))
        self.grasped = False
        self.contact_mode = "no_contact"

    def reset(self) -> dict[str, np.ndarray]:
        super().reset()
        self.grasped = False
        self.contact_mode = "no_contact"
        return self.observation()

    def target_position(self) -> np.ndarray:
        return self.goal

    def _advance(self, action: np.ndarray, controller: dict[str, float]) -> None:
        velocity_scale = float(np.clip(controller.get("velocity_scale", 1.0), 0.1, 2.0))
        damping = float(np.clip(controller.get("damping", 15.0), 1.0, 60.0))
        grip_force = float(np.clip(controller.get("grip_force", 18.0), 0.0, 60.0))
        desired_velocity = clip_norm(action[:3] * 0.7 * velocity_scale, 1.0)
        blend = float(np.clip(0.65 - damping / 150.0, 0.2, 0.62))
        self.ee_vel += blend * (desired_velocity - self.ee_vel)
        self.ee_pos += self.ee_vel * self.dt
        closing = action[3] > 0.0
        self.grip_width = np.float32(np.clip(self.grip_width - action[3] * 0.12, 0.0, 1.0))
        near = np.linalg.norm(self.ee_pos - self.obj_pos) < 0.16
        self.contact = bool(near and closing)
        friction_capacity = grip_force * max(self.physics.friction, 0.01) * 1.8
        required = self.physics.mass * 9.81
        if self.contact and friction_capacity >= required:
            self.grasped = True
        if self.grasped:
            slip_ratio = float(np.clip(required / max(friction_capacity, 1e-6) - 0.82, 0.0, 1.0))
            self.slipped = self.slipped or slip_ratio > 0.08
            self.contact_mode = "slip" if slip_ratio > 0.08 else "stick"
            follow_velocity = self.ee_vel.copy()
            follow_velocity[2] -= slip_ratio * 0.45
            self.obj_vel += 0.7 * (follow_velocity - self.obj_vel)
            self.obj_pos += self.obj_vel * self.dt
            self.contact_force[2] = grip_force
            if slip_ratio > 0.85:
                self.grasped = False
        else:
            self.contact_mode = "slip" if self.contact else "no_contact"
            self.contact_force[:] = 0
            if self.obj_pos[2] > 0.04:
                self.obj_vel[2] -= 9.81 * self.dt
                self.obj_pos += self.obj_vel * self.dt
                if self.obj_pos[2] <= 0.04:
                    self.obj_pos[2] = 0.04
                    self.obj_vel[:] = 0
                    self.dropped = self.t > 5

    def is_success(self) -> bool:
        return bool(self.grasped and np.linalg.norm(self.obj_pos - self.goal) <= self.tolerance)

    def metrics(self) -> dict[str, float | bool]:
        error = float(np.linalg.norm(self.obj_pos - self.goal))
        force = float(np.linalg.norm(self.contact_force))
        return {
            "success": self.is_success(),
            "final_error": error,
            "completion_time": self.t * self.dt,
            "contact_force": force,
            "drop": self.dropped,
            "slip": self.slipped,
            "force_violation": force > self.force_limit,
        }
