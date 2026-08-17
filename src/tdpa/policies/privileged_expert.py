"""Privileged nominal demonstrators used only to label robosuite trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExpertDecision:
    action: np.ndarray
    phase: str


class PrivilegedScriptedExpert:
    """Stateful Cartesian expert; simulator state must never enter a learned policy."""

    version = "tdpa-privileged-expert-v1"

    def __init__(self, task: str, *, position_delta_limit: float = 0.05) -> None:
        if task not in {"push", "lift"}:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.position_delta_limit = float(position_delta_limit)
        self.reset()

    def reset(self) -> None:
        self.phase = 0
        self.close_steps = 0
        self.grasp_losses = 0
        self._grasp_loss_active = False

    def _motion(self, goal: np.ndarray, eef: np.ndarray, *, limit: float = 1.0) -> np.ndarray:
        delta = (np.asarray(goal) - np.asarray(eef)) / self.position_delta_limit
        return np.clip(delta, -limit, limit).astype(np.float32)

    def act(self, env: Any, observation: dict[str, np.ndarray]) -> ExpertDecision:
        if set(observation) != {"rgbd", "proprio"}:
            raise RuntimeError("Expert labels require the exact deployment observation schema")
        object_position = np.asarray(
            env.raw.sim.data.body_xpos[env._body_id], dtype=np.float64
        ).copy()
        target_position = np.asarray(
            env.raw.sim.data.site_xpos[env._target_site_id], dtype=np.float64
        ).copy()
        eef_position = np.asarray(observation["proprio"][:3], dtype=np.float64)
        if self.task == "push":
            return self._push(object_position, target_position, eef_position)
        return self._lift(env, object_position, target_position, eef_position)

    def _push(
        self, object_position: np.ndarray, target_position: np.ndarray, eef_position: np.ndarray
    ) -> ExpertDecision:
        if self.phase == 0:
            phase = "approach_above"
            goal = object_position + np.array([-0.09, 0.0, 0.12])
            if np.linalg.norm(eef_position - goal) < 0.025:
                self.phase = 1
        elif self.phase == 1:
            phase = "descend_behind"
            goal = object_position + np.array([-0.07, 0.0, 0.015])
            if np.linalg.norm(eef_position - goal) < 0.022:
                self.phase = 2
        else:
            phase = "push"
            goal = target_position.copy()
            goal[0] += 0.04
            goal[2] = object_position[2] + 0.015
        limit = 0.7 if phase == "push" else 1.0
        action = np.concatenate([self._motion(goal, eef_position, limit=limit), [1.0]])
        return ExpertDecision(action.astype(np.float32), phase)

    def _lift(
        self,
        env: Any,
        object_position: np.ndarray,
        target_position: np.ndarray,
        eef_position: np.ndarray,
    ) -> ExpertDecision:
        grasped = bool(
            env.raw._check_grasp(
                gripper=env.raw.robots[0].gripper,
                object_geoms=env.raw.cube,
            )
        )
        if self.phase >= 3 and not grasped and not self._grasp_loss_active:
            self.grasp_losses += 1
            self._grasp_loss_active = True
        elif grasped:
            self._grasp_loss_active = False
        if self.phase == 0:
            phase = "approach_above"
            goal = object_position + np.array([0.0, 0.0, 0.12])
            grip = -1.0
            if np.linalg.norm(eef_position - goal) < 0.025:
                self.phase = 1
        elif self.phase == 1:
            phase = "descend_to_grasp"
            goal = object_position + np.array([0.0, 0.0, -0.003])
            grip = -1.0
            if np.linalg.norm(eef_position - goal) < 0.015:
                self.phase = 2
        elif self.phase == 2:
            phase = "close_gripper"
            goal = object_position + np.array([0.0, 0.0, -0.003])
            grip = 1.0
            self.close_steps += 1
            if self.close_steps >= 8 and grasped:
                self.phase = 3
        elif self.phase == 3:
            phase = "lift"
            goal = np.array([object_position[0], object_position[1], target_position[2]])
            grip = 1.0
            if abs(eef_position[2] - target_position[2]) < 0.025:
                self.phase = 4
        else:
            phase = "transport"
            goal = target_position
            grip = 1.0
        action = np.concatenate([self._motion(goal, eef_position), [grip]])
        return ExpertDecision(action.astype(np.float32), phase)
