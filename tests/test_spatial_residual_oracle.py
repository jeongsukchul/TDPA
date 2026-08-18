from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tdpa.controllers.spatial_residual_oracle import (
    PrivilegedSpatialResidualOracle,
    bounded_spatial_residual,
)
from tdpa.envs.base import Physics
from tdpa.evaluation.evaluate_nominal_policy import _json_hash
from tdpa.evaluation.robosuite_oracle_v2 import ORACLE_V2_GATE_VERSION, _make_manifest
from tdpa.evaluation.robosuite_spatial_residual_gate import (
    CELLS,
    _gate_decision,
    _reject_checkpoint_overlap,
    _validate_source_artifact,
)
from tdpa.utils.config import load_yaml


def test_spatial_residual_is_exactly_bounded_and_reports_projection() -> None:
    desired, bounded, projected, near_bound = bounded_spatial_residual(
        np.asarray([0.9, 0.0, 0.0], dtype=np.float32),
        np.asarray([-1.0, 0.5, 0.0], dtype=np.float32),
        velocity_scale=1.0,
        residual_limit=0.12,
        near_bound_fraction=0.95,
    )
    assert desired == pytest.approx([-1.9, 0.5, 0.0])
    assert bounded == pytest.approx([-0.12, 0.12, 0.0])
    assert projected
    assert near_bound

    with pytest.raises(ValueError, match="3-D"):
        bounded_spatial_residual(
            np.zeros(2),
            np.zeros(3),
            velocity_scale=1.0,
            residual_limit=0.12,
            near_bound_fraction=0.95,
        )
    with pytest.raises(ValueError, match="finite"):
        bounded_spatial_residual(
            np.asarray([np.nan, 0.0, 0.0]),
            np.zeros(3),
            velocity_scale=1.0,
            residual_limit=0.12,
            near_bound_fraction=0.95,
        )


def test_spatial_oracle_requires_explicit_privileged_permission() -> None:
    oracle = PrivilegedSpatialResidualOracle(
        load_yaml("configs/oracle/robosuite_spatial_residual_v1.yaml"),
        load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml"),
        load_yaml("configs/adapter/lift.yaml"),
        position_delta_limit=0.05,
        enabled=False,
    )
    observation = {
        "rgbd": np.zeros((4, 64, 64), dtype=np.float32),
        "proprio": np.zeros(10, dtype=np.float32),
    }
    with pytest.raises(PermissionError, match="enabled=True"):
        oracle.decide(
            None,
            Physics(1.0, 0.3),
            np.zeros(4, dtype=np.float32),
            observation,
        )


def _source_artifact() -> dict[str, object]:
    manifest = _make_manifest(
        task="lift",
        cells=CELLS,
        seeds=[4101, 4102, 4103],
        reset_index_start=40_000,
        episodes=20,
    )
    rows: list[dict[str, object]] = []
    for spec in manifest:
        fingerprint = f"reset-{spec['seed']}-{spec['episode']}"
        for method in ("b0", "perfect_context_oracle_v2"):
            rows.append(
                {
                    **spec,
                    "method": method,
                    "reset_fingerprint": fingerprint,
                    "success": False,
                    "final_error": 0.2,
                    "force_violation": False,
                    "saturation_rate": 0.0,
                }
            )
    return {
        "oracle_gate_version": ORACLE_V2_GATE_VERSION,
        "mode": "development",
        "status": "FAIL",
        "task": "lift",
        "checkpoint_sha256": "a" * 64,
        "environment_hash": "b" * 64,
        "competence_artifact_sha256": "c" * 64,
        "oracle_config_sha256": "d" * 64,
        "adapter_config_sha256": "e" * 64,
        "physics_ood_config_sha256": "f" * 64,
        "lift_friction_validation_sha256": "1" * 64,
        "oracle_revision": 3,
        "cells": list(CELLS),
        "seeds": [4101, 4102, 4103],
        "episodes_per_seed_cell": 20,
        "reset_index_start": 40_000,
        "methods": ["b0", "perfect_context_oracle_v2"],
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
        "gate": {
            "passed": False,
            "nominal_success_rate": 0.9,
            "cells": {
                "ood_mass_high": {"recovery_pass": False},
                "ood_friction_low": {"recovery_pass": True},
                "ood_composition": {"recovery_pass": True},
            },
        },
        "failures": [],
        "summaries": {},
        "rows": rows,
    }


def test_spatial_gate_accepts_only_exact_paired_failed_r3_source(tmp_path) -> None:
    artifact = _source_artifact()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded, digest = _validate_source_artifact(
        path,
        checkpoint_hash="a" * 64,
        environment_hash="b" * 64,
        competence_hash="c" * 64,
        oracle_config_hash="d" * 64,
        adapter_config_hash="e" * 64,
        physics_ood_config_hash="f" * 64,
        lift_friction_validation_hash="1" * 64,
    )
    assert loaded["status"] == "FAIL"
    assert len(digest) == 64

    artifact["rows"][1]["reset_fingerprint"] = "not-paired"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="not reset-paired"):
        _validate_source_artifact(
            path,
            checkpoint_hash="a" * 64,
            environment_hash="b" * 64,
            competence_hash="c" * 64,
            oracle_config_hash="d" * 64,
            adapter_config_hash="e" * 64,
            physics_ood_config_hash="f" * 64,
            lift_friction_validation_hash="1" * 64,
        )


def test_spatial_gate_requires_success_recovery_bounds_and_gripper_preservation() -> None:
    summaries: dict[str, dict[str, float]] = {}
    recovery: dict[str, dict[str, object]] = {}
    for cell in CELLS:
        summaries[f"b0/{cell}"] = {
            "success_rate": 0.1,
            "force_violation_rate": 0.0,
        }
        summaries[f"privileged_spatial_residual/{cell}"] = {
            "success_rate": 0.9,
            "force_violation_rate": 0.0,
            "mean_saturation_rate": 0.0,
            "maximum_residual_abs": 0.12,
            "mean_nominal_gripper_override_rate": 0.0,
        }
        recovery[cell] = {
            "success_recovery": 0.8,
            "success_recovery_bootstrap_95_interval": [0.6, 0.9],
            "reset_fingerprints_match": True,
        }
    thresholds = load_yaml("configs/oracle/robosuite_spatial_residual_v1.yaml")["gate"]
    assert _gate_decision(
        nominal_success=0.9,
        summaries=summaries,
        recovery=recovery,
        thresholds=thresholds,
    )["passed"]

    summaries["privileged_spatial_residual/ood_mass_high"]["success_rate"] = 0.79
    assert not _gate_decision(
        nominal_success=0.9,
        summaries=summaries,
        recovery=recovery,
        thresholds=thresholds,
    )["passed"]


def test_spatial_gate_rejects_checkpoint_reset_overlap() -> None:
    policy = SimpleNamespace(
        provenance={
            "collection_seed": 4101,
            "train_episode_ids": [40_000],
            "validation_episode_ids": [40_001],
        }
    )
    with pytest.raises(ValueError, match="overlaps checkpoint training"):
        _reject_checkpoint_overlap(
            policy,
            [{"seed": 4101, "reset_index": 40_001}],
        )
    _reject_checkpoint_overlap(policy, [{"seed": 4102, "reset_index": 40_001}])
