from __future__ import annotations

from typing import Any

from tdpa.envs.base import Physics, SyntheticManipulationEnv
from tdpa.envs.lift_env import LiftEnv
from tdpa.envs.push_env import PushEnv
from tdpa.utils.config import load_yaml


def make_env(
    task: str,
    *,
    physics: Physics | None = None,
    seed: int = 0,
    backend: str | None = None,
    config: dict[str, Any] | None = None,
) -> SyntheticManipulationEnv:
    config = config or load_yaml(f"configs/env/{task}.yaml")
    backend = backend or str(config.get("backend", "synthetic"))
    physics = physics or Physics(1.0, 0.55)
    if backend != "synthetic":
        if backend == "robosuite":
            raise RuntimeError(
                "The robosuite backend boundary is reserved but its task assets are not bundled in "
                "this dependency-free MVP. Install the simulation extra and provide audited task assets."
            )
        raise ValueError(f"Unknown backend: {backend}")
    env_type = {"push": PushEnv, "lift": LiftEnv}.get(task)
    if env_type is None:
        raise ValueError(f"Unknown task: {task}")
    return env_type(config, physics, seed)

