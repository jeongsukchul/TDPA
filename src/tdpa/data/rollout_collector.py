from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.controllers.oracle_context import OracleContextAdapter
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.policies.frozen_nominal import FrozenNominalPolicy
from tdpa.utils.config import load_yaml


@dataclass(frozen=True)
class AdapterExample:
    nominal_action: np.ndarray
    proprio: np.ndarray
    physics_context: np.ndarray
    target_residual: np.ndarray
    target_parameters: np.ndarray
    task_id: int


def collect_oracle_corrections(
    task: str, physics_values: list[Physics], seed: int = 0
) -> list[AdapterExample]:
    examples: list[AdapterExample] = []
    mapper = AdapterActionMapper(load_yaml(f"configs/adapter/{task}.yaml"))
    oracle = OracleContextAdapter(task, enabled=True)
    for episode_index, physics in enumerate(physics_values):
        env = make_env(task, physics=physics, seed=seed + episode_index)
        observation = env.reset()
        policy = FrozenNominalPolicy(task, env.config)
        for _ in range(env.horizon):
            nominal = policy(observation)
            correction = oracle.correction(physics)
            parameters = np.array(
                [
                    float(correction.get("velocity_scale", 1.0)),
                    float(correction.get("stiffness", 100.0)),
                    float(correction.get("damping", 15.0)),
                    float(correction.get("grip_force", 18.0)),
                ],
                dtype=np.float32,
            )
            examples.append(
                AdapterExample(
                    nominal_action=nominal.copy(),
                    proprio=observation["proprio"].copy(),
                    physics_context=np.array([physics.mass, physics.friction], dtype=np.float32),
                    target_residual=np.asarray(
                        correction.get("cartesian_residual", np.zeros(3)), dtype=np.float32
                    ),
                    target_parameters=parameters,
                    task_id=0 if task == "push" else 1,
                )
            )
            applied = mapper.apply(nominal, correction)
            observation, _, terminated, truncated, _ = env.step(applied.action, applied.controller)
            if terminated or truncated:
                break
    return examples

