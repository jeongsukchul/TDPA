"""Locked closed-loop competence and directional OOD evaluation for frozen policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import load_physics_config
from tdpa.envs.physics_randomization import PhysicsRandomizer, PhysicsSplit
from tdpa.policies.learned_nominal import (
    FrozenLearnedNominalPolicy,
    checkpoint_sha256,
    current_environment_hash,
)
from tdpa.utils.config import load_yaml

EVALUATION_VERSION = "tdpa-nominal-gate-v2"
COMPETENCE_SUCCESS_THRESHOLD = 0.8
EVALUATION_INDEX_START = 20_000
OOD_CELLS = (
    "nominal",
    "id",
    "ood_mass_low",
    "ood_mass_high",
    "ood_friction_low",
    "ood_friction_high",
    "ood_composition",
)
_CELL_CODES = {
    "ood_mass_low": 611,
    "ood_mass_high": 613,
    "ood_friction_low": 617,
    "ood_friction_high": 619,
}


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _physics_for_cell(cell: str, *, seed: int, episode: int) -> Physics:
    train = load_yaml("configs/physics/train.yaml")
    nominal = train["nominal"]
    if cell == "nominal":
        return Physics(float(nominal["mass"]), float(nominal["friction"]))
    config = load_physics_config()
    randomizer = PhysicsRandomizer(config, seed)
    if cell == "id":
        sample = randomizer.sample_at(PhysicsSplit.ID, episode)
        return Physics(sample.mass, sample.friction)
    if cell == "ood_composition":
        sample = randomizer.sample_at(PhysicsSplit.OOD_COMPOSITION, episode)
        return Physics(sample.mass, sample.friction)
    if cell not in _CELL_CODES:
        raise ValueError(f"Unknown evaluation cell: {cell}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _CELL_CODES[cell], episode]))
    if cell.startswith("ood_mass"):
        support = config.mass_ood.ranges[0 if cell.endswith("low") else -1]
        return Physics(support.sample(rng), config.friction_train.sample(rng))
    support = config.friction_ood.ranges[0 if cell.endswith("low") else -1]
    return Physics(config.mass_train.sample(rng), support.sample(rng))


def _bootstrap_interval(values: np.ndarray, *, seed: int = 0) -> list[float]:
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(2000, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _validate_competence_artifact(path: Path, *, task: str, checkpoint_hash: str) -> str:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "PASS" or artifact.get("mode") != "competence":
        raise ValueError("OOD evaluation requires a passing competence artifact")
    if artifact.get("task") != task or artifact.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Competence artifact task/checkpoint does not match OOD evaluation")
    if artifact.get("episodes_per_seed_cell") != 20 or len(artifact.get("seeds", [])) != 3:
        raise ValueError("Competence artifact does not use the locked 3x20 budget")
    if artifact.get("thresholds", {}).get("minimum_nominal_success") != 0.8:
        raise ValueError("Competence artifact threshold is not the locked 0.8")
    return _json_hash(artifact)


def _resolve_budget(args: argparse.Namespace) -> tuple[list[int], int, tuple[str, ...]]:
    if args.mode == "smoke":
        seeds = [11] if args.seeds is None else args.seeds
        episodes = 1 if args.episodes is None else args.episodes
        if len(seeds) != 1 or episodes != 1:
            raise ValueError("Smoke mode is locked to one seed and one episode per cell")
        return seeds, episodes, OOD_CELLS
    seeds = [11, 22, 33] if args.seeds is None else args.seeds
    episodes = 20 if args.episodes is None else args.episodes
    if len(seeds) != 3 or len(set(seeds)) != 3 or episodes != 20:
        raise ValueError("Competence/OOD modes require three distinct seeds and 20 episodes")
    return seeds, episodes, ("nominal",) if args.mode == "competence" else OOD_CELLS


def _rollout(
    *,
    task: str,
    cell: str,
    seed: int,
    episode: int,
    policy: FrozenLearnedNominalPolicy,
) -> dict[str, Any]:
    physics = _physics_for_cell(cell, seed=seed, episode=episode)
    reset_index = EVALUATION_INDEX_START + episode
    env = make_env(
        task,
        physics=physics,
        seed=seed,
        episode_index=reset_index,
        backend="robosuite",
    )
    policy.reset()
    observation_hash = hashlib.sha256()
    action_hash = hashlib.sha256()
    forces: list[float] = []
    action_boundary_steps = 0
    try:
        observation = env.reset()
        reset_fingerprint = env.reset_fingerprint()
        reset_state = env.reset_state()
        readback = env.read_physics()
        environment_hash = current_environment_hash(task)
        controller_hash = _json_hash(env.robosuite_config["execution"])
        info = env.metrics()
        steps = 0
        for steps in range(1, env.horizon + 1):
            for key in ("rgbd", "proprio"):
                observation_hash.update(np.ascontiguousarray(observation[key]).tobytes())
            action = policy.act(observation)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise RuntimeError("Frozen policy produced an invalid action")
            action_hash.update(np.ascontiguousarray(action).tobytes())
            action_boundary_steps += int(bool(np.any(np.abs(action[:3]) >= 1.0 - 1e-6)))
            observation, _, terminated, truncated, info = env.step(action)
            forces.append(float(info["contact_force"]))
            if terminated or truncated:
                break
        force_array = np.asarray(forces, dtype=np.float64)
        row: dict[str, Any] = {
            "task": task,
            "cell": cell,
            "seed": seed,
            "episode": episode,
            "reset_index": reset_index,
            "reset_fingerprint": reset_fingerprint,
            "reset_state": reset_state.as_dict(),
            "mass": physics.mass,
            "friction": physics.friction,
            "physics_readback": readback.as_dict(),
            "environment_hash": environment_hash,
            "controller_config_sha256": controller_hash,
            "control_frequency": int(env.robosuite_config["control_frequency"]),
            "horizon": env.horizon,
            "success": bool(info["success"]),
            "final_error": float(info["final_error"]),
            "completion_time": float(info["completion_time"]),
            "steps": steps,
            "peak_contact_force": float(force_array.max(initial=0.0)),
            "rms_contact_force": float(np.sqrt(np.mean(np.square(force_array)))),
            "force_violation": bool(force_array.max(initial=0.0) > env.force_limit),
            "action_boundary_rate": action_boundary_steps / steps,
            "observation_trace_sha256": observation_hash.hexdigest(),
            "action_trace_sha256": action_hash.hexdigest(),
            "controller_dictionary_used": False,
            "renderer": os.environ.get("MUJOCO_GL", "default"),
            "versions": env.versions(),
            "exception": None,
        }
        for key in ("overshoot", "drop", "slip"):
            if key in info:
                row[key] = info[key]
        return row
    finally:
        env.close()


def evaluate_nominal_policy(args: argparse.Namespace) -> dict[str, Any]:
    seeds, episodes, cells = _resolve_budget(args)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    competence_artifact_hash: str | None = None
    if args.mode == "ood":
        if args.competence_artifact is None:
            raise ValueError("--competence-artifact is required in OOD mode")
        competence_artifact_hash = _validate_competence_artifact(
            args.competence_artifact,
            task=args.task,
            checkpoint_hash=checkpoint_hash,
        )
    policy = FrozenLearnedNominalPolicy(
        args.checkpoint,
        task=args.task,
        device=args.device,
        allow_untrained=args.mode == "smoke",
    )
    trained_reset_ids = set(policy.provenance.get("train_episode_ids", [])) | set(
        policy.provenance.get("validation_episode_ids", [])
    )
    collection_seed = policy.provenance.get("collection_seed")
    for seed in seeds:
        evaluation_ids = {EVALUATION_INDEX_START + episode for episode in range(episodes)}
        if seed == collection_seed and trained_reset_ids & evaluation_ids:
            raise ValueError("Evaluation reset seed/index pairs overlap checkpoint training data")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fingerprints: dict[tuple[int, int], str] = {}
    for cell in cells:
        for seed in seeds:
            for episode in range(episodes):
                try:
                    row = _rollout(
                        task=args.task,
                        cell=cell,
                        seed=seed,
                        episode=episode,
                        policy=policy,
                    )
                    pair = (seed, episode)
                    expected = fingerprints.setdefault(pair, row["reset_fingerprint"])
                    if row["reset_fingerprint"] != expected:
                        raise RuntimeError("Paired physics cells produced different reset states")
                    rows.append(row)
                except Exception as error:  # noqa: BLE001 - preserve manifest attrition
                    failures.append(
                        {
                            "task": args.task,
                            "cell": cell,
                            "seed": seed,
                            "episode": episode,
                            "exception": f"{type(error).__name__}: {error}",
                        }
                    )

    summaries: dict[str, dict[str, float | int]] = {}
    for cell in cells:
        selected = [row for row in rows if row["cell"] == cell]
        summaries[cell] = {
            "episodes": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected]))
            if selected
            else 0.0,
            "mean_final_error": (
                float(np.mean([row["final_error"] for row in selected])) if selected else 0.0
            ),
            "force_violation_rate": (
                float(np.mean([row["force_violation"] for row in selected])) if selected else 0.0
            ),
        }
    paired_degradation: dict[str, dict[str, float | list[float]]] = {}
    per_seed_summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for cell in cells:
        per_seed_summaries[cell] = {}
        for seed in seeds:
            selected = [row for row in rows if row["cell"] == cell and row["seed"] == seed]
            per_seed_summaries[cell][str(seed)] = {
                "episodes": len(selected),
                "success_rate": (
                    float(np.mean([row["success"] for row in selected])) if selected else 0.0
                ),
                "mean_final_error": (
                    float(np.mean([row["final_error"] for row in selected])) if selected else 0.0
                ),
            }
    nominal_by_pair = {
        (row["seed"], row["episode"]): float(row["success"])
        for row in rows
        if row["cell"] == "nominal"
    }
    for cell in cells:
        if cell == "nominal":
            continue
        differences = np.asarray(
            [
                nominal_by_pair[(row["seed"], row["episode"])] - float(row["success"])
                for row in rows
                if row["cell"] == cell and (row["seed"], row["episode"]) in nominal_by_pair
            ],
            dtype=np.float64,
        )
        paired_degradation[cell] = {
            "mean": float(differences.mean()) if len(differences) else 0.0,
            "bootstrap_95_interval": _bootstrap_interval(differences),
        }

    expected_rows = len(cells) * len(seeds) * episodes
    complete = len(rows) == expected_rows and not failures
    nominal_rate = float(summaries["nominal"]["success_rate"])
    passed = complete and (
        args.mode != "competence" or nominal_rate >= COMPETENCE_SUCCESS_THRESHOLD
    )
    result = {
        "evaluation_version": EVALUATION_VERSION,
        "mode": args.mode,
        "status": "PASS" if passed else "FAIL",
        "scope": "frozen nominal policy competence and descriptive directional OOD sensitivity",
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "environment_hash": current_environment_hash(args.task),
        "competence_artifact_sha256": competence_artifact_hash,
        "manifest_sha256": _json_hash(
            [
                {key: row[key] for key in ("task", "cell", "seed", "episode", "mass", "friction")}
                for row in rows
            ]
            + failures
        ),
        "seeds": seeds,
        "episodes_per_seed_cell": episodes,
        "evaluation_index_start": EVALUATION_INDEX_START,
        "cells": list(cells),
        "thresholds": {"minimum_nominal_success": COMPETENCE_SUCCESS_THRESHOLD},
        "summaries": summaries,
        "per_seed_summaries": per_seed_summaries,
        "paired_success_degradation": paired_degradation,
        "failures": failures,
        "rows": rows,
        "force_definition": (
            "Push: object-table contact norm used only to audit the randomized friction interface. "
            "Lift: object-fingerpad contact norm. Neither is calibrated manipulation-force safety evidence."
        ),
        "warning": (
            "MuJoCo physics ranges and force units are uncalibrated. OOD output is descriptive; "
            "this does not establish adaptation or representation-learning claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise RuntimeError(f"Frozen nominal-policy {args.mode} gate failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "competence", "ood"), required=True)
    parser.add_argument("--task", choices=("push", "lift"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competence-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    evaluate_nominal_policy(build_parser().parse_args())


if __name__ == "__main__":
    main()
