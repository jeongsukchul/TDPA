from __future__ import annotations

import numpy as np
import pytest
import torch

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.models.physical_adapter import PhysicalAdapter
from tdpa.utils.config import load_yaml


@pytest.mark.parametrize("task", ["push", "lift"])
def test_neural_adapter_outputs_remain_inside_configured_bounds(task: str) -> None:
    config = load_yaml(f"configs/adapter/{task}.yaml")
    adapter = PhysicalAdapter(config)
    inputs = torch.full((16, 4), 1e9)
    latent = torch.full((16, 32), -1e9)
    proprio = torch.randn(16, 10) * 1e9
    output = adapter(inputs, latent, proprio)
    residual_limit = config["bounds"]["residual_max"]
    assert torch.all(output["cartesian_residual"].abs() <= residual_limit)
    for name in ("velocity_scale", "stiffness", "damping", "grip_force"):
        low, high = config["bounds"][name]
        assert torch.all(output[name] >= low)
        assert torch.all(output[name] <= high)


def test_controller_side_mapper_sanitizes_nonfinite_and_extreme_outputs() -> None:
    mapper = AdapterActionMapper(load_yaml("configs/adapter/lift.yaml"))
    applied = mapper.apply(
        np.array([np.inf, -np.inf, np.nan, 5.0], dtype=np.float32),
        {
            "cartesian_residual": np.array([np.inf, -999.0, np.nan]),
            "velocity_scale": np.inf,
            "stiffness": -1e20,
            "damping": 1e20,
            "grip_force": np.nan,
        },
    )
    assert np.isfinite(applied.action).all()
    assert applied.execution_command.shape == (7,)
    assert np.isfinite(applied.execution_command).all()
    assert np.all(np.abs(applied.action) <= 1.0)
    assert 50.0 <= applied.controller["stiffness"] <= 220.0
    assert 5.0 <= applied.controller["damping"] <= 35.0
    assert 3.0 <= applied.controller["grip_force"] <= 60.0
    assert applied.saturated


def test_controller_targets_are_part_of_execution_command() -> None:
    mapper = AdapterActionMapper(load_yaml("configs/adapter/lift.yaml"))
    action = np.array([0.2, 0.0, 0.1, 1.0], dtype=np.float32)
    low = mapper.apply(action, {"grip_force": 8.0, "damping": 6.0, "stiffness": 60.0})
    high = mapper.apply(action, {"grip_force": 50.0, "damping": 30.0, "stiffness": 200.0})
    assert np.array_equal(low.action, high.action)
    assert not np.array_equal(low.execution_command, high.execution_command)


def test_isolated_nonfinite_inputs_are_reported_as_saturation() -> None:
    mapper = AdapterActionMapper(load_yaml("configs/adapter/lift.yaml"))
    finite = np.zeros(4, dtype=np.float32)
    assert mapper.apply(np.array([np.nan, 0.0, 0.0, 0.0])).saturated
    assert mapper.apply(finite, {"velocity_scale": np.nan}).saturated
    assert mapper.apply(finite, {"cartesian_residual": [0.0, np.inf, 0.0]}).saturated
    assert mapper.apply(finite, {"grip_force": -np.inf}).saturated
    assert mapper.apply(np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)).saturated
