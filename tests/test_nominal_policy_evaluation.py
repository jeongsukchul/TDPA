from __future__ import annotations

import argparse
import json

import pytest

from tdpa.evaluation.evaluate_nominal_policy import (
    _physics_for_cell,
    _resolve_budget,
    _validate_competence_artifact,
)


def arguments(mode: str, *, seeds=None, episodes=None) -> argparse.Namespace:
    return argparse.Namespace(mode=mode, seeds=seeds, episodes=episodes)


def test_evaluation_modes_lock_their_budgets_and_cells() -> None:
    seeds, episodes, cells = _resolve_budget(arguments("competence"))
    assert seeds == [11, 22, 33]
    assert episodes == 20
    assert cells == ("nominal",)
    with pytest.raises(ValueError, match="three distinct seeds"):
        _resolve_budget(arguments("competence", seeds=[11], episodes=1))
    smoke_seeds, smoke_episodes, smoke_cells = _resolve_budget(arguments("smoke"))
    assert smoke_seeds == [11]
    assert smoke_episodes == 1
    assert "ood_mass_low" in smoke_cells and "ood_mass_high" in smoke_cells


def test_low_and_high_ood_cells_are_directionally_disjoint() -> None:
    low_mass = _physics_for_cell("ood_mass_low", seed=3, episode=4)
    high_mass = _physics_for_cell("ood_mass_high", seed=3, episode=4)
    low_friction = _physics_for_cell("ood_friction_low", seed=3, episode=4)
    high_friction = _physics_for_cell("ood_friction_high", seed=3, episode=4)
    assert 0.25 <= low_mass.mass < 0.45
    assert 1.8 <= high_mass.mass < 2.4
    assert 0.08 <= low_friction.friction < 0.22
    assert 0.9 <= high_friction.friction < 1.2


def test_lift_uses_calibrated_low_friction_without_changing_push() -> None:
    push = _physics_for_cell("ood_friction_low", seed=3, episode=4, task="push")
    lift = _physics_for_cell("ood_friction_low", seed=3, episode=4, task="lift")
    assert 0.08 <= push.friction < 0.22
    assert 0.29 <= lift.friction < 0.34
    assert push.mass == lift.mass


def test_ood_requires_matching_locked_competence_artifact(tmp_path) -> None:
    path = tmp_path / "competence.json"
    artifact = {
        "status": "PASS",
        "mode": "competence",
        "task": "push",
        "checkpoint_sha256": "a" * 64,
        "episodes_per_seed_cell": 20,
        "seeds": [11, 22, 33],
        "thresholds": {"minimum_nominal_success": 0.8},
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert len(_validate_competence_artifact(path, task="push", checkpoint_hash="a" * 64)) == 64
    with pytest.raises(ValueError, match="does not match"):
        _validate_competence_artifact(path, task="lift", checkpoint_hash="a" * 64)
