from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.controllers.oracle_context import RobosuitePerfectContextOracle
from tdpa.envs.base import Physics
from tdpa.evaluation.robosuite_oracle_gate import (
    _gate_decision,
    _paired_recovery,
    _source_rows,
    _summaries,
    _validate_execution_contract,
)
from tdpa.utils.config import load_yaml


def _oracle(task: str, *, enabled: bool = True) -> RobosuitePerfectContextOracle:
    return RobosuitePerfectContextOracle(
        task,
        load_yaml("configs/oracle/robosuite_perfect_context.yaml"),
        load_yaml(f"configs/adapter/{task}.yaml"),
        enabled=enabled,
    )


def test_oracle_requires_explicit_privileged_access() -> None:
    with pytest.raises(PermissionError, match="requires enabled=True"):
        _oracle("push", enabled=False).correction(Physics(1.0, 0.55))


@pytest.mark.parametrize("task", ["push", "lift"])
def test_oracle_commands_are_finite_and_within_adapter_bounds(task: str) -> None:
    adapter = load_yaml(f"configs/adapter/{task}.yaml")
    oracle = _oracle(task)
    for physics in (Physics(0.25, 0.08), Physics(1.0, 0.55), Physics(2.4, 1.2)):
        command = oracle.correction(physics)
        assert np.isfinite(command["cartesian_residual"]).all()
        assert np.max(np.abs(command["cartesian_residual"])) <= adapter["bounds"]["residual_max"]
        for key in ("velocity_scale", "stiffness", "damping", "grip_force"):
            low, high = adapter["bounds"][key]
            assert low <= command[key] <= high


def test_oracle_schedule_has_expected_direction_without_cell_labels() -> None:
    push = _oracle("push")
    push_nominal = push.correction(Physics(1.0, 0.55))
    push_high_friction = push.correction(Physics(1.0, 1.1))
    assert push_high_friction["velocity_scale"] > push_nominal["velocity_scale"]
    assert push_high_friction["stiffness"] > push_nominal["stiffness"]

    lift = _oracle("lift")
    lift_nominal = lift.correction(Physics(1.0, 0.55))
    lift_high_mass = lift.correction(Physics(2.0, 0.55))
    lift_low_friction = lift.correction(Physics(1.0, 0.11))
    assert lift_high_mass["velocity_scale"] < lift_nominal["velocity_scale"]
    assert lift_high_mass["stiffness"] > lift_nominal["stiffness"]
    assert lift_high_mass["damping"] > lift_nominal["damping"]
    assert lift_high_mass["grip_force"] > lift_nominal["grip_force"]
    assert lift_low_friction["velocity_scale"] < lift_nominal["velocity_scale"]
    assert lift_low_friction["grip_force"] > lift_nominal["grip_force"]


def test_oracle_command_uses_the_same_bounded_mapper_as_learned_adapters() -> None:
    adapter_config = load_yaml("configs/adapter/lift.yaml")
    mapper = AdapterActionMapper(adapter_config)
    command = _oracle("lift").correction(Physics(2.4, 0.08))
    applied = mapper.apply(np.asarray([0.8, -0.6, 0.4, 1.0], dtype=np.float32), command)
    assert applied.action.shape == (4,)
    assert np.isfinite(applied.execution_command).all()
    assert np.max(np.abs(applied.action)) <= 1.0
    for key in ("stiffness", "damping", "grip_force"):
        low, high = adapter_config["bounds"][key]
        assert low <= applied.controller[key] <= high


def test_adapter_and_live_execution_contract_must_match() -> None:
    adapter_config = load_yaml("configs/adapter/push.yaml")
    mapper = AdapterActionMapper(adapter_config)
    execution = {
        "nominal": dict(adapter_config["nominal"]),
        "bounds": {
            key: list(adapter_config["bounds"][key])
            for key in ("velocity_scale", "stiffness", "damping", "grip_force")
        },
    }
    env = SimpleNamespace(robosuite_config={"execution": execution})
    _validate_execution_contract(env, mapper)
    execution["bounds"]["stiffness"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="bounds for stiffness"):
        _validate_execution_contract(env, mapper)


def test_source_manifest_requires_complete_unique_locked_rows() -> None:
    source = {
        "cells": ["ood_mass_high"],
        "seeds": [11, 22, 33],
        "rows": [
            {"cell": "ood_mass_high", "seed": seed, "episode": episode}
            for seed in (11, 22, 33)
            for episode in range(20)
        ],
    }
    assert len(_source_rows(source, cells=("ood_mass_high",), mode="full")) == 60
    source["rows"][-1] = dict(source["rows"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        _source_rows(source, cells=("ood_mass_high",), mode="full")


def _paired_rows(*, oracle_successes: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in range(10):
        for method, successes, error in (
            ("b0", 1, 0.2),
            ("perfect_context_oracle", oracle_successes, 0.05),
        ):
            rows.append(
                {
                    "method": method,
                    "cell": "ood_mass_high",
                    "seed": 11,
                    "episode": episode,
                    "success": episode < successes,
                    "final_error": error,
                    "force_violation": False,
                    "saturation_rate": 0.0,
                    "source_b0_success_reproduced": True,
                }
            )
    return rows


def test_gate_requires_material_paired_recovery_within_bounds() -> None:
    cells = ("ood_mass_high",)
    source = {"summaries": {"nominal": {"success_rate": 0.9}}}
    config = load_yaml("configs/oracle/robosuite_perfect_context.yaml")["gate"]
    passing_rows = _paired_rows(oracle_successes=7)
    passing_summary = _summaries(passing_rows, cells)
    passing_recovery = _paired_recovery(passing_rows, cells)
    passing = _gate_decision(
        source=source,
        rows=passing_rows,
        summaries=passing_summary,
        recovery=passing_recovery,
        cells=cells,
        gate_config=config,
    )
    assert passing["passed"]
    assert passing["cells"]["ood_mass_high"]["absolute_success_recovery"] == pytest.approx(0.6)

    failing_rows = _paired_rows(oracle_successes=2)
    failing = _gate_decision(
        source=source,
        rows=failing_rows,
        summaries=_summaries(failing_rows, cells),
        recovery=_paired_recovery(failing_rows, cells),
        cells=cells,
        gate_config=config,
    )
    assert not failing["passed"]
    assert not failing["cells"]["ood_mass_high"]["recovery_pass"]
