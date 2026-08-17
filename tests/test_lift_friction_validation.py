from __future__ import annotations

import json

import pytest

from tdpa.envs.physics_config import load_physics_config, validate_lift_physics_activation
from tdpa.evaluation.evaluate_nominal_policy import _json_hash
from tdpa.evaluation.lift_friction_calibration import REFINEMENT_VERSION
from tdpa.evaluation.lift_friction_validation import (
    _gate_decision,
    _make_manifest,
    _resolve_protocol,
    _validate_candidate_ood,
    _validate_refinement_artifact,
)
from tdpa.utils.config import load_yaml


def _config():
    return load_yaml("configs/evaluation/lift_friction_validation.yaml")


def test_lift_candidate_support_is_task_specific_and_disjoint_from_train() -> None:
    candidate = _validate_candidate_ood(_config())
    assert candidate["status"] == "candidate_pending_held_out_validation"
    lift = load_physics_config(ood_path=_config()["candidate_ood_config"])
    original = load_physics_config()
    assert [lift.friction_ood.ranges[0].low, lift.friction_ood.ranges[0].high] == [0.29, 0.34]
    assert [original.friction_ood.ranges[0].low, original.friction_ood.ranges[0].high] == [
        0.08,
        0.22,
    ]
    assert not lift.friction_ood.ranges[0].overlaps(lift.friction_train)


def test_validation_manifest_is_locked_deterministic_and_fresh() -> None:
    config = _config()
    seeds, uniform_start, uniform_episodes, boundary_start, boundary_episodes = _resolve_protocol(
        config, "validation"
    )
    assert seeds == [7301, 7302, 7303]
    assert set(seeds).isdisjoint({7101, 7102, 7103, 7201, 7202, 7203})
    manifest = _make_manifest(
        config=config,
        seeds=seeds,
        uniform_start=uniform_start,
        uniform_episodes=uniform_episodes,
        boundary_start=boundary_start,
        boundary_episodes=boundary_episodes,
    )
    assert manifest == _make_manifest(
        config=config,
        seeds=seeds,
        uniform_start=uniform_start,
        uniform_episodes=uniform_episodes,
        boundary_start=boundary_start,
        boundary_episodes=boundary_episodes,
    )
    assert len(manifest) == 75
    uniform = [row for row in manifest if row["cell"] == "calibrated_uniform"]
    boundary = [row for row in manifest if row["cell"] == "boundary_stress"]
    assert len(uniform) == 60
    assert len(boundary) == 15
    assert all(0.60 <= row["mass"] < 1.40 and 0.29 <= row["friction"] < 0.34 for row in uniform)
    assert all(row["mass"] == 1.40 and row["friction"] == 0.29 for row in boundary)


def test_validation_requires_the_exact_passing_refinement(tmp_path) -> None:
    environment_hash = "environment"
    refinement_config_hash = _json_hash(
        load_yaml("configs/evaluation/lift_friction_refinement.yaml")
    )
    artifact = {
        "calibration_version": REFINEMENT_VERSION,
        "calibration_stage": "refinement",
        "mode": "development",
        "status": "PASS",
        "environment_hash": environment_hash,
        "config_sha256": refinement_config_hash,
        "failures": [],
        "gate": {
            "paired_resets_pass": True,
            "recommended_low_friction_support": [0.29, 0.34],
            "levels": [
                {"friction": friction, "all_masses_feasible": True}
                for friction in (0.29, 0.30, 0.31, 0.32, 0.33, 0.34)
            ],
        },
    }
    source = tmp_path / "refinement.json"
    source.write_text(json.dumps(artifact), encoding="utf-8")
    assert _validate_refinement_artifact(
        source,
        environment_hash=environment_hash,
        refinement_config_hash=refinement_config_hash,
    ) == _json_hash(artifact)

    artifact["gate"]["levels"][-1]["all_masses_feasible"] = False
    source.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="every calibrated friction"):
        _validate_refinement_artifact(
            source,
            environment_hash=environment_hash,
            refinement_config_hash=refinement_config_hash,
        )


def _summary(success: float, lower: float, *, force: float = 0.0) -> dict[str, object]:
    return {
        "success_rate": success,
        "success_bootstrap_95_interval": [lower, 1.0],
        "force_violation_rate": force,
        "mean_controller_saturation_rate": 0.0,
    }


def test_validation_gate_requires_uniform_and_boundary_support() -> None:
    summaries = {
        "calibrated_uniform": _summary(0.9, 0.8),
        "boundary_stress": _summary(0.8, 0.6),
    }
    assert _gate_decision(summaries, _config()["gate"])["passed"]
    summaries["boundary_stress"] = _summary(0.8, 0.6, force=0.2)
    assert not _gate_decision(summaries, _config()["gate"])["passed"]


def test_activation_requires_complete_held_out_validation_artifact(tmp_path) -> None:
    environment_hash = "environment"
    manifest = [{"cell": "calibrated_uniform", "seed": 7301}]
    artifact = {
        "validation_version": "tdpa-lift-friction-validation-v1",
        "mode": "validation",
        "status": "PASS",
        "task": "lift",
        "environment_hash": environment_hash,
        "config_sha256": _json_hash(_config()),
        "candidate_ood_config_sha256": _json_hash(
            load_yaml("configs/physics/ood_lift_calibrated_v1.yaml")
        ),
        "seeds": [7301, 7302, 7303],
        "uniform_episodes_per_seed": 20,
        "boundary_episodes_per_seed": 5,
        "source_refinement_sha256": "a" * 64,
        "failures": [],
        "summaries": {
            "calibrated_uniform": {"episodes": 60},
            "boundary_stress": {"episodes": 15},
        },
        "gate": {
            "passed": True,
            "cells": {
                "calibrated_uniform": {"passed": True},
                "boundary_stress": {"passed": True},
            },
        },
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
    }
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert len(validate_lift_physics_activation(path, environment_hash=environment_hash)) == 64

    artifact["summaries"]["boundary_stress"]["episodes"] = 14
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete boundary"):
        validate_lift_physics_activation(path, environment_hash=environment_hash)
