from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tdpa.envs.physics_randomization import PhysicsRandomizationConfig
from tdpa.utils.config import load_yaml

DEFAULT_OOD_PATH = "configs/physics/ood.yaml"
LIFT_CALIBRATED_OOD_PATH = "configs/physics/ood_lift_calibrated_v1.yaml"


def physics_ood_path(task: str | None = None) -> str:
    if task not in {None, "push", "lift"}:
        raise ValueError(f"Unsupported task for physics configuration: {task}")
    return LIFT_CALIBRATED_OOD_PATH if task == "lift" else DEFAULT_OOD_PATH


def load_physics_config(
    train_path: str = "configs/physics/train.yaml",
    ood_path: str | None = None,
    *,
    task: str | None = None,
) -> PhysicsRandomizationConfig:
    if ood_path is not None and task is not None:
        raise ValueError("Specify either ood_path or task, not both")
    ood_path = ood_path or physics_ood_path(task)
    train = load_yaml(train_path)
    ood = load_yaml(ood_path)
    return PhysicsRandomizationConfig.from_mappings(train, ood)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_lift_physics_activation(path: Path, *, environment_hash: str) -> str:
    """Validate the held-out PASS required to activate Lift's calibrated OOD support."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    validation_config = load_yaml("configs/evaluation/lift_friction_validation.yaml")
    candidate_config = load_yaml(LIFT_CALIBRATED_OOD_PATH)
    expected = {
        "validation_version": "tdpa-lift-friction-validation-v1",
        "mode": "validation",
        "status": "PASS",
        "task": "lift",
        "environment_hash": environment_hash,
        "config_sha256": _json_hash(validation_config),
        "candidate_ood_config_sha256": _json_hash(candidate_config),
        "seeds": [7301, 7302, 7303],
        "uniform_episodes_per_seed": 20,
        "boundary_episodes_per_seed": 5,
    }
    mismatched = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatched:
        raise ValueError(
            f"Lift OOD activation requires the matching held-out PASS: {sorted(mismatched)}"
        )
    source_hash = artifact.get("source_refinement_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("Lift OOD activation artifact has no refinement provenance")
    if artifact.get("failures") or not artifact.get("gate", {}).get("passed"):
        raise ValueError("Lift OOD activation artifact has failures or a failed gate")
    summaries = artifact.get("summaries", {})
    if summaries.get("calibrated_uniform", {}).get("episodes") != 60:
        raise ValueError("Lift OOD activation artifact has an incomplete uniform sample")
    if summaries.get("boundary_stress", {}).get("episodes") != 15:
        raise ValueError("Lift OOD activation artifact has an incomplete boundary sample")
    if not all(
        artifact.get("gate", {}).get("cells", {}).get(cell, {}).get("passed")
        for cell in ("calibrated_uniform", "boundary_stress")
    ):
        raise ValueError("Lift OOD activation artifact did not pass both support cells")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, list) or artifact.get("manifest_sha256") != _json_hash(manifest):
        raise ValueError("Lift OOD activation artifact manifest is invalid")
    return hashlib.sha256(path.read_bytes()).hexdigest()
