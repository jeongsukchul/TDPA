from __future__ import annotations

import pytest

from tdpa.envs.base import GeomPhysicsReadback, Physics, PhysicsReadback
from tdpa.evaluation.lift_feasibility import (
    _controller_for_phase,
    _failure_stage,
    _gate_decision,
    _make_manifest,
    _resolve_protocol,
    _validate_physics_readback,
    _validate_profiles,
)
from tdpa.utils.config import load_yaml


def _config():
    return load_yaml("configs/evaluation/lift_feasibility.yaml")


def test_feasibility_manifest_is_locked_and_disjoint_from_oracle_splits() -> None:
    cells, seeds, start, episodes = _resolve_protocol(_config(), "development")
    assert cells == ("ood_mass_high", "ood_friction_low")
    assert seeds == [6101, 6102, 6103]
    assert set(seeds).isdisjoint({4101, 4102, 4103, 5101, 5102, 5103})
    assert start == 70_000
    assert episodes == 20
    manifest = _make_manifest(
        cells=cells,
        seeds=seeds,
        index_start=start,
        episodes=episodes,
    )
    assert len(manifest) == 120
    assert {row["cell"] for row in manifest} == set(cells)
    assert all(row["reset_index"] >= 70_000 for row in manifest)


def test_all_feasibility_profiles_are_inside_live_execution_bounds() -> None:
    config = _config()
    execution = load_yaml("configs/env/lift.yaml")["robosuite"]["execution"]
    profiles = _validate_profiles(config["controller_profiles"], execution)
    assert profiles["nominal"] == {key: float(value) for key, value in execution["nominal"].items()}
    assert profiles["high_grip"]["grip_force"] == execution["bounds"]["grip_force"][1]
    assert profiles["high_authority"]["stiffness"] == execution["bounds"]["stiffness"][1]
    gentle_lift = _controller_for_phase(profiles["gentle_lift"], "lift")
    gentle_approach = _controller_for_phase(profiles["gentle_lift"], "approach_above")
    assert gentle_lift["velocity_scale"] == 0.5
    assert gentle_approach["velocity_scale"] == 1.0

    invalid = {name: dict(profile) for name, profile in config["controller_profiles"].items()}
    invalid["high_authority"]["stiffness"] = 221.0
    with pytest.raises(ValueError, match="exceeds execution bounds"):
        _validate_profiles(invalid, execution)


def _summary(success: float, lower: float, *, force: float = 0.0) -> dict[str, object]:
    return {
        "success_rate": success,
        "success_bootstrap_95_interval": [lower, min(1.0, success + 0.1)],
        "force_violation_rate": force,
        "mean_controller_saturation_rate": 0.0,
    }


def test_feasibility_gate_requires_one_safe_passing_profile_per_cell() -> None:
    config = _config()
    profiles = _validate_profiles(
        config["controller_profiles"],
        load_yaml("configs/env/lift.yaml")["robosuite"]["execution"],
    )
    cells = ("ood_mass_high", "ood_friction_low")
    summaries = {f"{profile}/{cell}": _summary(0.4, 0.25) for profile in profiles for cell in cells}
    summaries["high_authority/ood_mass_high"] = _summary(0.9, 0.8)
    summaries["high_grip/ood_friction_low"] = _summary(0.85, 0.7)
    gate = _gate_decision(
        summaries,
        cells=cells,
        profiles=profiles,
        thresholds=config["gate"],
        paired_resets_pass=True,
    )
    assert gate["passed"]
    assert gate["cells"]["ood_mass_high"]["best_profile"] == "high_authority"
    assert gate["cells"]["ood_friction_low"]["best_profile"] == "high_grip"

    summaries["high_grip/ood_friction_low"] = _summary(0.85, 0.7, force=0.2)
    assert not _gate_decision(
        summaries,
        cells=cells,
        profiles=profiles,
        thresholds=config["gate"],
        paired_resets_pass=True,
    )["passed"]


def _readback(*, mass: float = 0.8, friction: float = 0.05) -> PhysicsReadback:
    geom = GeomPhysicsReadback("cube", 1, (friction, 0.005, 0.0001))
    pad = GeomPhysicsReadback("fingerpad", 2, (friction, 0.005, 0.0001))
    return PhysicsReadback(
        backend="robosuite",
        requested_mass=mass,
        requested_friction=friction,
        actual_mass=mass,
        body_name="cube_main",
        body_id=1,
        body_inertia=(1.0, 1.0, 1.0),
        object_geoms=(geom,),
        counterpart_geoms=(pad,),
        topology_signature="test",
    )


def test_feasibility_requires_exact_live_physics_readback() -> None:
    _validate_physics_readback(_readback(), Physics(0.8, 0.05))
    with pytest.raises(RuntimeError, match="mass readback"):
        _validate_physics_readback(_readback(mass=0.81), Physics(0.8, 0.05))
    with pytest.raises(RuntimeError, match="friction readback"):
        _validate_physics_readback(_readback(friction=0.06), Physics(0.8, 0.05))


@pytest.mark.parametrize(
    ("success", "contact", "grasped", "final_grasped", "losses", "expected"),
    [
        (True, 0, 0, False, 0, None),
        (False, 0, 0, False, 0, "no_fingerpad_contact"),
        (False, 5, 0, False, 0, "contact_without_grasp"),
        (False, 5, 3, False, 0, "grasp_lost"),
        (False, 5, 3, True, 1, "grasp_lost"),
        (False, 5, 3, True, 0, "lift_or_transport_timeout"),
    ],
)
def test_failure_stage_classification(
    success: bool,
    contact: int,
    grasped: int,
    final_grasped: bool,
    losses: int,
    expected: str | None,
) -> None:
    assert (
        _failure_stage(
            success=success,
            contact_steps=contact,
            grasped_steps=grasped,
            final_grasped=final_grasped,
            grasp_losses=losses,
        )
        == expected
    )
