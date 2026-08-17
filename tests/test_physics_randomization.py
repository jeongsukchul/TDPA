from __future__ import annotations

import pytest

from tdpa.envs.physics_randomization import (
    ParameterRange,
    ParameterSupport,
    PhysicsRandomizationConfig,
    PhysicsRandomizer,
    PhysicsSample,
    PhysicsSplit,
)


@pytest.fixture()
def config() -> PhysicsRandomizationConfig:
    return PhysicsRandomizationConfig(
        mass_train=(0.5, 1.5),
        friction_train=(0.3, 0.9),
        mass_ood=(1.5, 3.0),
        friction_ood=(0.05, 0.3),
        composition_mass=(1.2, 1.5),
        composition_friction=(0.3, 0.45),
        pretraining_policy_id="scripted_v1",
        shifted_policy_id="scripted_v2",
    )


def test_ranges_are_half_open_and_adjacent_ranges_do_not_overlap() -> None:
    left = ParameterRange(0.0, 1.0)
    right = ParameterRange(1.0, 2.0)
    assert left.contains(0.0)
    assert not left.contains(1.0)
    assert not left.overlaps(right)


def test_disjoint_support_rejects_internal_overlap() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        ParameterSupport((ParameterRange(0.1, 0.5), ParameterRange(0.4, 0.8)))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mass_ood": (1.4, 2.0)}, "mass_ood"),
        ({"friction_ood": (0.2, 0.4)}, "friction_ood"),
        ({"composition_mass": (1.4, 1.6)}, "composition_mass"),
        ({"composition_friction": (0.2, 0.4)}, "composition_friction"),
    ],
)
def test_invalid_or_overlapping_supports_fail_loudly(
    override: dict[str, tuple[float, float]], message: str
) -> None:
    kwargs = {
        "mass_train": (0.5, 1.5),
        "friction_train": (0.3, 0.9),
        "mass_ood": (1.5, 3.0),
        "friction_ood": (0.05, 0.3),
        "composition_mass": (1.2, 1.5),
        "composition_friction": (0.3, 0.45),
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        PhysicsRandomizationConfig(**kwargs)


def test_sampling_is_deterministic_and_independent_of_call_order(
    config: PhysicsRandomizationConfig,
) -> None:
    first = PhysicsRandomizer(config, seed=17)
    second = PhysicsRandomizer(config, seed=17)
    for split in PhysicsSplit:
        assert first.sample(split, 20) == second.sample(split, 20)

    ordered = PhysicsRandomizer(config, seed=29)
    expected = ordered.sample_at(PhysicsSplit.OOD_MASS, 7)
    ordered.sample(PhysicsSplit.TRAIN, 100)
    ordered.sample(PhysicsSplit.OOD_FRICTION, 50)
    assert ordered.sample_at(PhysicsSplit.OOD_MASS, 7) == expected


def test_all_split_samples_respect_support_and_composition_is_withheld(
    config: PhysicsRandomizationConfig,
) -> None:
    randomizer = PhysicsRandomizer(config, seed=123)
    for split in PhysicsSplit:
        for sample in randomizer.sample(split, 500):
            config.validate_sample(sample)

    train = randomizer.sample(PhysicsSplit.TRAIN, 500)
    assert not any(config.is_composition_holdout(x.mass, x.friction) for x in train)
    composition = randomizer.sample(PhysicsSplit.OOD_COMPOSITION, 500)
    assert all(config.is_composition_holdout(x.mass, x.friction) for x in composition)


def test_ood_axes_change_exactly_the_declared_marginal(
    config: PhysicsRandomizationConfig,
) -> None:
    randomizer = PhysicsRandomizer(config, seed=5)
    for sample in randomizer.sample(PhysicsSplit.OOD_MASS, 100):
        assert config.mass_ood.contains(sample.mass)
        assert config.friction_train.contains(sample.friction)
    for sample in randomizer.sample(PhysicsSplit.OOD_FRICTION, 100):
        assert config.mass_train.contains(sample.mass)
        assert config.friction_ood.contains(sample.friction)


def test_policy_shift_pairs_identical_physics_with_a_new_policy(
    config: PhysicsRandomizationConfig,
) -> None:
    randomizer = PhysicsRandomizer(config, seed=91)
    for index in range(100):
        identity = randomizer.sample_at(PhysicsSplit.ID, index)
        shifted = randomizer.sample_at(PhysicsSplit.POLICY_SHIFT, index)
        assert (identity.mass, identity.friction) == (shifted.mass, shifted.friction)
        assert identity.behavior_policy_id == config.pretraining_policy_id
        assert shifted.behavior_policy_id == config.shifted_policy_id
        assert identity.behavior_policy_id != shifted.behavior_policy_id


def test_declared_split_rejects_wrong_behavior_policy(
    config: PhysicsRandomizationConfig,
) -> None:
    with pytest.raises(ValueError, match="sample policy"):
        config.validate_sample(
            PhysicsSample(
                mass=0.7,
                friction=0.7,
                split=PhysicsSplit.POLICY_SHIFT,
                behavior_policy_id=config.pretraining_policy_id,
                sample_index=0,
            )
        )


def test_different_seeds_change_samples(config: PhysicsRandomizationConfig) -> None:
    left = PhysicsRandomizer(config, seed=1).sample(PhysicsSplit.TRAIN, 10)
    right = PhysicsRandomizer(config, seed=2).sample(PhysicsSplit.TRAIN, 10)
    assert [(x.mass, x.friction) for x in left] != [(x.mass, x.friction) for x in right]


def test_two_sided_ood_support_and_repository_mapping_are_supported() -> None:
    train = {"mass": {"train": [0.6, 1.4]}, "friction": {"train": [0.35, 0.75]}}
    ood = {
        "mass": {"low": [0.25, 0.45], "high": [1.8, 2.4]},
        "friction": {"low": [0.08, 0.22], "high": [0.9, 1.2]},
        "composition": {"mass": [1.3, 1.4], "friction": [0.35, 0.4]},
        "policy_shift": {"behavior_policy": "chirp"},
    }
    mapped = PhysicsRandomizationConfig.from_mappings(train, ood)
    samples = PhysicsRandomizer(mapped, seed=77).sample(PhysicsSplit.OOD_MASS, 500)
    assert any(sample.mass < mapped.mass_train.low for sample in samples)
    assert any(sample.mass > mapped.mass_train.high for sample in samples)
    assert all(mapped.mass_ood.contains(sample.mass) for sample in samples)
    assert mapped.shifted_policy_id == "chirp"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mass_train", (-0.1, 1.0)),
        ("mass_ood", ((-1.0, -0.5), (1.5, 2.0))),
        ("friction_train", (-0.1, 0.5)),
        ("friction_ood", ((-0.5, -0.2), (0.9, 1.2))),
    ],
)
def test_unphysical_parameter_support_is_rejected(field: str, value: object) -> None:
    kwargs = {
        "mass_train": (0.5, 1.5),
        "friction_train": (0.3, 0.9),
        "mass_ood": (1.5, 3.0),
        "friction_ood": (0.05, 0.3),
        "composition_mass": (1.2, 1.5),
        "composition_friction": (0.3, 0.45),
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        PhysicsRandomizationConfig(**kwargs)
