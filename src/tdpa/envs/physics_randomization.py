"""Deterministic mass/friction randomization with explicit OOD supports.

The split definitions in this module are deliberately about *support*, not
about a finite set of pre-generated values.  Ranges are half-open intervals,
which means adjacent intervals such as ``[0.5, 1.0)`` and ``[1.0, 2.0)`` are
disjoint.  This convention avoids a boundary value silently belonging to both
train and OOD.

OOD-composition is a rectangular region inside the marginal training ranges
that is withheld from train/ID sampling.  Policy-shift samples are paired with
ID samples by index: their physics is identical and only the behavior-policy
label differs.  This makes policy-shift comparisons controlled rather than
confounding policy and physics.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Any

import numpy as np


class PhysicsSplit(str, Enum):
    """Supported randomization/evaluation splits."""

    TRAIN = "train"
    ID = "id"
    OOD_MASS = "ood_mass"
    OOD_FRICTION = "ood_friction"
    OOD_COMPOSITION = "ood_composition"
    POLICY_SHIFT = "policy_shift"


@dataclass(frozen=True)
class ParameterRange:
    """A finite, non-empty half-open interval ``[low, high)``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        low = float(self.low)
        high = float(self.high)
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError("range bounds must be finite")
        if low >= high:
            raise ValueError(f"range must satisfy low < high, got [{low}, {high})")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= float(value) < self.high

    def contains_range(self, other: ParameterRange) -> bool:
        return self.low <= other.low and other.high <= self.high

    def overlaps(self, other: ParameterRange) -> bool:
        """Return whether two half-open intervals share positive support."""

        return max(self.low, other.low) < min(self.high, other.high)

    def sample(self, rng: np.random.Generator) -> float:
        # Generator.uniform follows the half-open convention (apart from
        # floating-point rounding documented by NumPy).
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class ParameterSupport:
    """A non-empty union of pairwise-disjoint half-open intervals."""

    ranges: tuple[ParameterRange, ...]

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError("parameter support must contain at least one range")
        ranges = tuple(sorted(self.ranges, key=lambda item: item.low))
        for left, right in pairwise(ranges):
            if left.overlaps(right):
                raise ValueError("ranges within one parameter support must not overlap")
        object.__setattr__(self, "ranges", ranges)

    @property
    def width(self) -> float:
        return sum(item.width for item in self.ranges)

    def contains(self, value: float) -> bool:
        return any(item.contains(value) for item in self.ranges)

    def overlaps(self, other: ParameterRange | ParameterSupport) -> bool:
        other_ranges = other.ranges if isinstance(other, ParameterSupport) else (other,)
        return any(left.overlaps(right) for left in self.ranges for right in other_ranges)

    def sample(self, rng: np.random.Generator) -> float:
        """Sample uniformly over the union, weighted by interval width."""

        draw = float(rng.uniform(0.0, self.width))
        cumulative = 0.0
        selected = self.ranges[-1]
        for item in self.ranges:
            cumulative += item.width
            if draw < cumulative:
                selected = item
                break
        return selected.sample(rng)


RangeInput = ParameterRange | tuple[float, float]
SupportInput = ParameterSupport | RangeInput | Sequence[RangeInput]


def _as_range(value: RangeInput, name: str) -> ParameterRange:
    if isinstance(value, ParameterRange):
        return value
    try:
        low, high = value
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be ParameterRange or a (low, high) pair") from error
    return ParameterRange(low, high)


def _as_support(value: SupportInput, name: str) -> ParameterSupport:
    if isinstance(value, ParameterSupport):
        return value
    if isinstance(value, ParameterRange):
        return ParameterSupport((value,))
    if (
        isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(bound, (int, float, np.number)) for bound in value)
    ):
        return ParameterSupport((_as_range(value, name),))
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a range or sequence of ranges")
    return ParameterSupport(
        tuple(_as_range(item, f"{name}[{index}]") for index, item in enumerate(value))
    )


@dataclass(frozen=True)
class PhysicsRandomizationConfig:
    """Validated split supports and behavior-policy identities.

    ``composition_mass`` and ``composition_friction`` jointly define the
    high-mass/low-friction (or other configured) rectangle withheld from
    TRAIN and ID.  Each must lie inside its corresponding training marginal.
    """

    mass_train: RangeInput
    friction_train: RangeInput
    mass_ood: SupportInput
    friction_ood: SupportInput
    composition_mass: RangeInput
    composition_friction: RangeInput
    pretraining_policy_id: str = "probe_train"
    shifted_policy_id: str = "probe_shift"

    def __post_init__(self) -> None:
        range_names = (
            "mass_train",
            "friction_train",
            "composition_mass",
            "composition_friction",
        )
        for name in range_names:
            object.__setattr__(self, name, _as_range(getattr(self, name), name))
        object.__setattr__(self, "mass_ood", _as_support(self.mass_ood, "mass_ood"))
        object.__setattr__(self, "friction_ood", _as_support(self.friction_ood, "friction_ood"))

        if self.mass_ood.overlaps(self.mass_train):
            raise ValueError("mass_ood must not overlap mass_train")
        if self.friction_ood.overlaps(self.friction_train):
            raise ValueError("friction_ood must not overlap friction_train")
        if not self.mass_train.contains_range(self.composition_mass):
            raise ValueError("composition_mass must be contained in mass_train")
        if not self.friction_train.contains_range(self.composition_friction):
            raise ValueError("composition_friction must be contained in friction_train")
        if (
            self.composition_mass == self.mass_train
            and self.composition_friction == self.friction_train
        ):
            raise ValueError("composition holdout cannot cover the entire train support")
        if self.mass_train.low <= 0 or any(item.low <= 0 for item in self.mass_ood.ranges):
            raise ValueError("mass ranges must be strictly positive")
        if self.friction_train.low < 0 or any(item.low < 0 for item in self.friction_ood.ranges):
            raise ValueError("friction ranges must be non-negative")
        if not self.pretraining_policy_id or not self.shifted_policy_id:
            raise ValueError("behavior policy IDs must be non-empty")
        if self.pretraining_policy_id == self.shifted_policy_id:
            raise ValueError("policy-shift evaluation requires a distinct behavior policy ID")

    @classmethod
    def from_mappings(
        cls,
        train: Mapping[str, Any],
        ood: Mapping[str, Any],
        *,
        pretraining_policy_id: str = "probe_train",
    ) -> PhysicsRandomizationConfig:
        """Build from the repository's ``train.yaml``/``ood.yaml`` structure."""

        try:
            mass_ood = tuple(ood["mass"][side] for side in ("low", "high"))
            friction_ood = tuple(ood["friction"][side] for side in ("low", "high"))
            shifted_policy_id = str(ood["policy_shift"]["behavior_policy"])
            return cls(
                mass_train=train["mass"]["train"],
                friction_train=train["friction"]["train"],
                mass_ood=mass_ood,
                friction_ood=friction_ood,
                composition_mass=ood["composition"]["mass"],
                composition_friction=ood["composition"]["friction"],
                pretraining_policy_id=pretraining_policy_id,
                shifted_policy_id=shifted_policy_id,
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid train/OOD physics configuration mapping") from error

    def is_composition_holdout(self, mass: float, friction: float) -> bool:
        return self.composition_mass.contains(mass) and self.composition_friction.contains(friction)

    def validate_sample(self, sample: PhysicsSample) -> None:
        """Raise if a sample violates its declared split contract."""

        mass = sample.mass
        friction = sample.friction
        split = sample.split
        if split in (PhysicsSplit.TRAIN, PhysicsSplit.ID, PhysicsSplit.POLICY_SHIFT):
            valid = self.mass_train.contains(mass) and self.friction_train.contains(friction)
            valid = valid and not self.is_composition_holdout(mass, friction)
        elif split is PhysicsSplit.OOD_MASS:
            valid = self.mass_ood.contains(mass) and self.friction_train.contains(friction)
        elif split is PhysicsSplit.OOD_FRICTION:
            valid = self.mass_train.contains(mass) and self.friction_ood.contains(friction)
        elif split is PhysicsSplit.OOD_COMPOSITION:
            valid = self.is_composition_holdout(mass, friction)
        else:  # pragma: no cover - exhaustive guard for future enum additions
            raise ValueError(f"unsupported split: {split}")
        if not valid:
            raise ValueError(f"sample ({mass}, {friction}) violates split {split.value}")
        expected_policy = (
            self.shifted_policy_id
            if split is PhysicsSplit.POLICY_SHIFT
            else self.pretraining_policy_id
        )
        if sample.behavior_policy_id != expected_policy:
            raise ValueError(
                f"sample policy {sample.behavior_policy_id!r} violates split {split.value}"
            )


@dataclass(frozen=True)
class PhysicsSample:
    """One physics configuration plus non-policy-input diagnostic metadata."""

    mass: float
    friction: float
    split: PhysicsSplit
    behavior_policy_id: str
    sample_index: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "mass": self.mass,
            "friction": self.friction,
            "split": self.split.value,
            "behavior_policy_id": self.behavior_policy_id,
            "sample_index": self.sample_index,
        }


# Fixed integer codes are used instead of Python's randomized string hash.
_SPLIT_SEED_CODE = {
    PhysicsSplit.TRAIN: 101,
    PhysicsSplit.ID: 211,
    PhysicsSplit.OOD_MASS: 307,
    PhysicsSplit.OOD_FRICTION: 401,
    PhysicsSplit.OOD_COMPOSITION: 503,
    # Policy shift intentionally reuses the ID physics stream.
    PhysicsSplit.POLICY_SHIFT: 211,
}


class PhysicsRandomizer:
    """Index-addressable deterministic sampler.

    ``sample_at`` is independent of call order.  ``sample`` is a convenience
    stateful iterator whose counter is separate for every split, so sampling
    TRAIN cannot perturb a subsequent OOD sequence.
    """

    def __init__(self, config: PhysicsRandomizationConfig, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if not 0 <= int(seed) < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        self.config = config
        self.seed = int(seed)
        self._next_index = {split: 0 for split in PhysicsSplit}

    @staticmethod
    def _coerce_split(split: PhysicsSplit | str) -> PhysicsSplit:
        try:
            return PhysicsSplit(split)
        except ValueError as error:
            choices = ", ".join(item.value for item in PhysicsSplit)
            raise ValueError(
                f"unknown physics split {split!r}; expected one of {choices}"
            ) from error

    def _rng(self, split: PhysicsSplit, sample_index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            entropy=[self.seed, _SPLIT_SEED_CODE[split], sample_index]
        )
        return np.random.default_rng(sequence)

    def _sample_train_support(self, rng: np.random.Generator) -> tuple[float, float]:
        """Sample uniformly from the train rectangle minus composition holdout.

        Decomposing the complement into disjoint rectangles avoids an
        unbounded rejection loop when a deliberately difficult test config
        makes the held-out rectangle nearly as large as the training support.
        """

        mass = self.config.mass_train
        friction = self.config.friction_train
        held_mass = self.config.composition_mass
        held_friction = self.config.composition_friction
        rectangles = [
            (mass.low, held_mass.low, friction.low, friction.high),
            (held_mass.high, mass.high, friction.low, friction.high),
            (held_mass.low, held_mass.high, friction.low, held_friction.low),
            (held_mass.low, held_mass.high, held_friction.high, friction.high),
        ]
        weighted = [
            (bounds, (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]))
            for bounds in rectangles
            if bounds[0] < bounds[1] and bounds[2] < bounds[3]
        ]
        total_area = sum(area for _, area in weighted)
        draw = float(rng.uniform(0.0, total_area))
        cumulative = 0.0
        selected = weighted[-1][0]
        for bounds, area in weighted:
            cumulative += area
            if draw < cumulative:
                selected = bounds
                break
        mass_low, mass_high, friction_low, friction_high = selected
        return (
            float(rng.uniform(mass_low, mass_high)),
            float(rng.uniform(friction_low, friction_high)),
        )

    def sample_at(self, split: PhysicsSplit | str, sample_index: int) -> PhysicsSample:
        split = self._coerce_split(split)
        if isinstance(sample_index, bool) or not isinstance(sample_index, (int, np.integer)):
            raise TypeError("sample_index must be an integer")
        if int(sample_index) < 0:
            raise ValueError("sample_index must be non-negative")
        sample_index = int(sample_index)
        rng = self._rng(split, sample_index)

        if split in (PhysicsSplit.TRAIN, PhysicsSplit.ID, PhysicsSplit.POLICY_SHIFT):
            mass, friction = self._sample_train_support(rng)
        elif split is PhysicsSplit.OOD_MASS:
            mass = self.config.mass_ood.sample(rng)
            friction = self.config.friction_train.sample(rng)
        elif split is PhysicsSplit.OOD_FRICTION:
            mass = self.config.mass_train.sample(rng)
            friction = self.config.friction_ood.sample(rng)
        elif split is PhysicsSplit.OOD_COMPOSITION:
            mass = self.config.composition_mass.sample(rng)
            friction = self.config.composition_friction.sample(rng)
        else:  # pragma: no cover - exhaustive guard
            raise AssertionError(split)

        behavior_policy_id = (
            self.config.shifted_policy_id
            if split is PhysicsSplit.POLICY_SHIFT
            else self.config.pretraining_policy_id
        )
        result = PhysicsSample(
            mass=mass,
            friction=friction,
            split=split,
            behavior_policy_id=behavior_policy_id,
            sample_index=sample_index,
        )
        self.config.validate_sample(result)
        return result

    def sample(self, split: PhysicsSplit | str, count: int = 1) -> list[PhysicsSample]:
        split = self._coerce_split(split)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise TypeError("count must be an integer")
        if int(count) < 0:
            raise ValueError("count must be non-negative")
        start = self._next_index[split]
        result = [self.sample_at(split, start + offset) for offset in range(int(count))]
        self._next_index[split] += int(count)
        return result

    def iter_split(self, split: PhysicsSplit | str) -> Iterator[PhysicsSample]:
        split = self._coerce_split(split)
        while True:
            yield self.sample(split, 1)[0]

    def reset(self, split: PhysicsSplit | str | None = None) -> None:
        """Reset one split counter, or all counters when ``split`` is None."""

        if split is None:
            for key in self._next_index:
                self._next_index[key] = 0
            return
        self._next_index[self._coerce_split(split)] = 0
