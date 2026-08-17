from __future__ import annotations

from tdpa.evaluation.lift_feasibility import _validate_profiles
from tdpa.evaluation.lift_friction_calibration import (
    _frontier_decision,
    _make_manifest,
    _resolve_protocol,
)
from tdpa.utils.config import load_yaml


def _config():
    return load_yaml("configs/evaluation/lift_friction_calibration.yaml")


def test_calibration_grid_is_locked_and_uses_a_fresh_namespace() -> None:
    masses, frictions, seeds, start, episodes = _resolve_protocol(_config(), "development")
    assert masses == [0.6, 0.9, 1.2, 1.4]
    assert frictions == [0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
    assert seeds == [7101, 7102, 7103]
    assert set(seeds).isdisjoint({4101, 4102, 4103, 5101, 5102, 5103, 6101, 6102, 6103})
    assert start == 90_000
    assert episodes == 5
    manifest = _make_manifest(
        masses=masses,
        frictions=frictions,
        seeds=seeds,
        index_start=start,
        episodes=episodes,
    )
    assert len(manifest) == 480
    assert len({(row["mass"], row["friction"]) for row in manifest}) == 32
    assert {row["reset_index"] for row in manifest} == set(range(90_000, 90_005))


def test_calibration_uses_locked_maximum_grip_profile_below_train_support() -> None:
    config = _config()
    env_config = load_yaml("configs/env/lift.yaml")
    feasibility = load_yaml("configs/evaluation/lift_feasibility.yaml")
    profiles = _validate_profiles(
        feasibility["controller_profiles"], env_config["robosuite"]["execution"]
    )
    name = config["protocol"]["controller_profile"]
    assert name == "high_grip"
    assert (
        profiles[name]["grip_force"]
        == env_config["robosuite"]["execution"]["bounds"]["grip_force"][1]
    )
    assert max(config["protocol"]["friction_grid"]) < config["gate"]["train_friction_lower_bound"]


def _summary(mass: float, friction: float, *, success: float, lower: float) -> dict[str, object]:
    return {
        "mass": mass,
        "friction": friction,
        "success_rate": success,
        "success_bootstrap_95_interval": [lower, min(1.0, success + 0.1)],
        "force_violation_rate": 0.0,
        "mean_controller_saturation_rate": 0.0,
    }


def test_frontier_requires_a_contiguous_suffix_feasible_at_every_mass() -> None:
    masses = [0.6, 1.4]
    frictions = [0.20, 0.22, 0.24]
    summaries = [
        _summary(mass, friction, success=0.9, lower=0.7)
        for mass in masses
        for friction in frictions
    ]
    summaries[0] = _summary(0.6, 0.20, success=0.5, lower=0.3)
    gate = _frontier_decision(
        summaries,
        masses=masses,
        frictions=frictions,
        thresholds=_config()["gate"],
        paired_resets_pass=True,
    )
    assert gate["passed"]
    assert gate["recommended_low_friction_support"] == [0.22, 0.24]

    summaries[-1]["force_violation_rate"] = 0.2
    gate = _frontier_decision(
        summaries,
        masses=masses,
        frictions=frictions,
        thresholds=_config()["gate"],
        paired_resets_pass=True,
    )
    assert not gate["passed"]
    assert gate["recommended_low_friction_support"] is None


def test_frontier_rejects_unpaired_resets() -> None:
    masses = [0.6, 1.4]
    frictions = [0.22, 0.24]
    summaries = [
        _summary(mass, friction, success=0.9, lower=0.7)
        for mass in masses
        for friction in frictions
    ]
    gate = _frontier_decision(
        summaries,
        masses=masses,
        frictions=frictions,
        thresholds=_config()["gate"],
        paired_resets_pass=False,
    )
    assert not gate["passed"]
    assert gate["recommended_low_friction_support"] == [0.22, 0.24]
