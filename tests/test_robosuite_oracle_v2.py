from __future__ import annotations

import json

import numpy as np
import pytest

from tdpa.controllers.oracle_context import RobosuitePerfectContextOracleV2
from tdpa.envs.base import Physics
from tdpa.evaluation.evaluate_nominal_policy import _json_hash
from tdpa.evaluation.robosuite_oracle_v2 import (
    DEFAULT_CELLS,
    ORACLE_V2_GATE_VERSION,
    _command_projected,
    _gate_decision,
    _make_manifest,
    _stage_protocol,
    _validate_development_artifact,
)
from tdpa.utils.config import load_yaml


def _observation(*, gripper_width: float = 0.08) -> dict[str, np.ndarray]:
    proprio = np.zeros(10, dtype=np.float32)
    proprio[6] = gripper_width
    return {
        "rgbd": np.zeros((4, 64, 64), dtype=np.float32),
        "proprio": proprio,
    }


def _oracle(task: str, *, enabled: bool = True) -> RobosuitePerfectContextOracleV2:
    return RobosuitePerfectContextOracleV2(
        task,
        load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml"),
        load_yaml(f"configs/adapter/{task}.yaml"),
        enabled=enabled,
    )


def test_oracle_v2_requires_explicit_physics_permission_and_rejects_leakage() -> None:
    action = np.asarray([0.0, 0.0, 0.2, 1.0], dtype=np.float32)
    with pytest.raises(PermissionError, match="requires enabled=True"):
        _oracle("lift", enabled=False).decide(Physics(2.0, 0.2), action, _observation())

    leaked = _observation()
    leaked["mass"] = np.asarray([2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="Privileged"):
        _oracle("lift").decide(Physics(2.0, 0.2), action, leaked)


def test_push_v2_limits_velocity_but_preserves_stiffness_compensation() -> None:
    oracle = _oracle("push")
    decision = oracle.decide(
        Physics(1.3, 1.2),
        np.asarray([0.9, 0.0, 0.0, -1.0], dtype=np.float32),
        _observation(),
    )
    assert 1.0 < decision.correction["velocity_scale"] <= 1.1
    assert decision.correction["stiffness"] > 100.0
    assert np.array_equal(decision.correction["cartesian_residual"], np.zeros(3))
    assert _command_projected(decision)


def test_lift_v2_preserves_timing_and_adds_only_causal_mass_residual() -> None:
    oracle = _oracle("lift")
    approach = oracle.decide(
        Physics(2.0, 0.11),
        np.asarray([0.1, 0.0, -0.2, -1.0], dtype=np.float32),
        _observation(),
    )
    assert approach.phase == "approach"
    assert approach.correction["velocity_scale"] == 1.0
    assert approach.correction["cartesian_residual"][2] == 0.0

    lift = oracle.decide(
        Physics(2.0, 0.11),
        np.asarray([0.0, 0.0, 0.5, 1.0], dtype=np.float32),
        _observation(),
    )
    assert lift.phase == "lift"
    assert lift.correction["velocity_scale"] == 1.0
    assert 0.0 < lift.correction["cartesian_residual"][2] <= 0.10
    assert lift.correction["grip_force"] > 18.0

    oracle.reset()
    light = oracle.decide(
        Physics(0.8, 0.55),
        np.asarray([0.0, 0.0, 0.5, 1.0], dtype=np.float32),
        _observation(gripper_width=0.04),
    )
    assert light.phase == "lift"
    assert light.correction["cartesian_residual"][2] == 0.0


def test_development_and_final_manifests_are_disjoint_and_locked() -> None:
    config = load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")
    dev_seeds, dev_start, episodes = _stage_protocol(config, "development")
    final_seeds, final_start, final_episodes = _stage_protocol(config, "final")
    assert set(dev_seeds).isdisjoint(final_seeds)
    assert dev_start != final_start
    assert episodes == final_episodes == 20

    dev = _make_manifest(
        task="lift",
        cells=DEFAULT_CELLS["lift"],
        seeds=dev_seeds,
        reset_index_start=dev_start,
        episodes=episodes,
    )
    final = _make_manifest(
        task="lift",
        cells=DEFAULT_CELLS["lift"],
        seeds=final_seeds,
        reset_index_start=final_start,
        episodes=final_episodes,
    )
    assert len(dev) == len(final) == 180
    assert {
        (row["seed"], row["reset_index"]) for row in dev
    }.isdisjoint({(row["seed"], row["reset_index"]) for row in final})


def test_final_requires_matching_passing_development_artifact(tmp_path) -> None:
    manifest = [{"seed": 4101, "episode": 0}]
    artifact = {
        "oracle_gate_version": ORACLE_V2_GATE_VERSION,
        "mode": "development",
        "status": "PASS",
        "task": "push",
        "checkpoint_sha256": "a" * 64,
        "environment_hash": "b" * 64,
        "oracle_config_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "seeds": [4101, 4102, 4103],
        "cells": list(DEFAULT_CELLS["push"]),
        "episodes_per_seed_cell": 20,
        "failures": [],
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
    }
    path = tmp_path / "development.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert len(
        _validate_development_artifact(
            path,
            task="push",
            checkpoint_hash="a" * 64,
            environment_hash="b" * 64,
            oracle_config_hash="c" * 64,
            adapter_config_hash="d" * 64,
        )
    ) == 64
    artifact["status"] = "FAIL"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="matching passing"):
        _validate_development_artifact(
            path,
            task="push",
            checkpoint_hash="a" * 64,
            environment_hash="b" * 64,
            oracle_config_hash="c" * 64,
            adapter_config_hash="d" * 64,
        )


def test_v2_gate_requires_pairing_recovery_bounds_and_force_diagnostic() -> None:
    summaries = {
        "b0/ood_mass_high": {
            "success_rate": 0.1,
            "force_violation_rate": 0.0,
        },
        "perfect_context_oracle_v2/ood_mass_high": {
            "success_rate": 0.7,
            "force_violation_rate": 0.05,
            "mean_saturation_rate": 0.02,
        },
    }
    recovery = {
        "ood_mass_high": {
            "success_recovery": 0.6,
            "success_recovery_bootstrap_95_interval": [0.4, 0.75],
            "reset_fingerprints_match": True,
        }
    }
    decision = _gate_decision(
        nominal_success=0.9,
        summaries=summaries,
        recovery=recovery,
        cells=("ood_mass_high",),
        gate_config=load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")["gate"],
    )
    assert decision["passed"]
    recovery["ood_mass_high"]["reset_fingerprints_match"] = False
    assert not _gate_decision(
        nominal_success=0.9,
        summaries=summaries,
        recovery=recovery,
        cells=("ood_mass_high",),
        gate_config=load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")["gate"],
    )["passed"]
