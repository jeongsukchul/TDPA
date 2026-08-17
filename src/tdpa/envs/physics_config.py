from __future__ import annotations

from tdpa.envs.physics_randomization import PhysicsRandomizationConfig
from tdpa.utils.config import load_yaml


def load_physics_config(
    train_path: str = "configs/physics/train.yaml",
    ood_path: str = "configs/physics/ood.yaml",
) -> PhysicsRandomizationConfig:
    train = load_yaml(train_path)
    ood = load_yaml(ood_path)
    return PhysicsRandomizationConfig.from_mappings(train, ood)
