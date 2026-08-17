"""Held-out validation for the candidate Lift-specific low-friction support."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.envs.physics_config import load_physics_config
from tdpa.evaluation.evaluate_nominal_policy import _bootstrap_interval, _json_hash
from tdpa.evaluation.lift_feasibility import _run_episode, _validate_profiles
from tdpa.evaluation.lift_friction_calibration import REFINEMENT_VERSION
from tdpa.policies.learned_nominal import current_environment_hash
from tdpa.policies.privileged_expert import PrivilegedScriptedExpert
from tdpa.utils.config import load_yaml

VALIDATION_VERSION = "tdpa-lift-friction-validation-v1"
CELLS = ("calibrated_uniform", "boundary_stress")


def _resolve_protocol(config: dict[str, Any], mode: str) -> tuple[list[int], int, int, int, int]:
    protocol = config["protocol"]
    if [float(value) for value in protocol["mass_support"]] != [0.60, 1.40]:
        raise ValueError("Lift validation mass support is not locked")
    if [float(value) for value in protocol["friction_support"]] != [0.29, 0.34]:
        raise ValueError("Lift validation friction support is not locked")
    if float(protocol["boundary_mass"]) != 1.40 or float(protocol["boundary_friction"]) != 0.29:
        raise ValueError("Lift validation boundary stress point is not locked")
    if mode == "smoke":
        return [31], 115_000, 1, 115_100, 1
    seeds = [int(seed) for seed in protocol["seeds"]]
    uniform_start = int(protocol["uniform_reset_index_start"])
    uniform_episodes = int(protocol["uniform_episodes_per_seed"])
    boundary_start = int(protocol["boundary_reset_index_start"])
    boundary_episodes = int(protocol["boundary_episodes_per_seed"])
    if (
        seeds != [7301, 7302, 7303]
        or uniform_start != 110_000
        or uniform_episodes != 20
        or boundary_start != 110_100
        or boundary_episodes != 5
    ):
        raise ValueError("Lift held-out validation protocol is not locked")
    return seeds, uniform_start, uniform_episodes, boundary_start, boundary_episodes


def _make_manifest(
    *,
    config: dict[str, Any],
    seeds: list[int],
    uniform_start: int,
    uniform_episodes: int,
    boundary_start: int,
    boundary_episodes: int,
) -> list[dict[str, Any]]:
    protocol = config["protocol"]
    mass_low, mass_high = map(float, protocol["mass_support"])
    friction_low, friction_high = map(float, protocol["friction_support"])
    manifest: list[dict[str, Any]] = []
    for seed in seeds:
        for episode in range(uniform_episodes):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 827, episode]))
            manifest.append(
                {
                    "task": "lift",
                    "cell": CELLS[0],
                    "seed": seed,
                    "episode": episode,
                    "reset_index": uniform_start + episode,
                    "mass": float(rng.uniform(mass_low, mass_high)),
                    "friction": float(rng.uniform(friction_low, friction_high)),
                }
            )
        for episode in range(boundary_episodes):
            manifest.append(
                {
                    "task": "lift",
                    "cell": CELLS[1],
                    "seed": seed,
                    "episode": episode,
                    "reset_index": boundary_start + episode,
                    "mass": float(protocol["boundary_mass"]),
                    "friction": float(protocol["boundary_friction"]),
                }
            )
    return manifest


def _validate_refinement_artifact(
    path: Path, *, environment_hash: str, refinement_config_hash: str
) -> str:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("calibration_version") != REFINEMENT_VERSION
        or artifact.get("calibration_stage") != "refinement"
        or artifact.get("mode") != "development"
        or artifact.get("status") != "PASS"
    ):
        raise ValueError("Held-out validation requires a passing refinement artifact")
    if artifact.get("environment_hash") != environment_hash:
        raise ValueError("Refinement environment does not match the live environment")
    if artifact.get("config_sha256") != refinement_config_hash:
        raise ValueError("Refinement artifact does not match the locked configuration")
    if artifact.get("failures") or not artifact.get("gate", {}).get("paired_resets_pass"):
        raise ValueError("Refinement artifact has rollout attrition or unpaired resets")
    if artifact.get("gate", {}).get("recommended_low_friction_support") != [0.29, 0.34]:
        raise ValueError("Refinement artifact did not recommend the candidate support")
    levels = artifact.get("gate", {}).get("levels", [])
    if len(levels) != 6 or not all(level.get("all_masses_feasible") for level in levels):
        raise ValueError("Refinement artifact does not pass every calibrated friction level")
    return _json_hash(artifact)


def _validate_candidate_ood(config: dict[str, Any]) -> dict[str, Any]:
    path = str(config["candidate_ood_config"])
    raw = load_yaml(path)
    if (
        raw.get("version") != "tdpa-lift-ood-calibrated-v1"
        or raw.get("task") != "lift"
        or raw.get("status") != "candidate_pending_held_out_validation"
    ):
        raise ValueError("Unsupported candidate Lift OOD configuration")
    physics = load_physics_config(ood_path=path)
    low = physics.friction_ood.ranges[0]
    if [low.low, low.high] != [0.29, 0.34]:
        raise ValueError("Candidate OOD configuration does not encode the calibrated support")
    if low.overlaps(physics.friction_train):
        raise ValueError("Candidate low-friction support overlaps training")
    return raw


def _summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        selected = [row for row in rows if row["cell"] == cell]
        successes = np.asarray([float(row["success"]) for row in selected], dtype=np.float64)
        failure_stages = Counter(
            str(row["failure_stage"]) for row in selected if row["failure_stage"] is not None
        )
        summaries[cell] = {
            "episodes": len(selected),
            "success_rate": float(successes.mean()),
            "success_bootstrap_95_interval": _bootstrap_interval(successes),
            "mean_final_error": float(np.mean([row["final_error"] for row in selected])),
            "force_violation_rate": float(np.mean([row["force_violation"] for row in selected])),
            "mean_controller_saturation_rate": float(
                np.mean([row["controller_saturation_rate"] for row in selected])
            ),
            "failure_stages": dict(sorted(failure_stages.items())),
        }
    return summaries


def _gate_decision(
    summaries: dict[str, dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for cell in CELLS:
        summary = summaries[cell]
        prefix = "uniform" if cell == CELLS[0] else "boundary"
        success_rate = float(summary["success_rate"])
        ci_lower = float(summary["success_bootstrap_95_interval"][0])
        passed = (
            success_rate >= float(thresholds[f"{prefix}_minimum_success_rate"])
            and ci_lower >= float(thresholds[f"{prefix}_minimum_success_ci_lower"])
            and float(summary["force_violation_rate"])
            <= float(thresholds["maximum_force_violation_rate"])
            and float(summary["mean_controller_saturation_rate"])
            <= float(thresholds["maximum_controller_saturation_rate"])
        )
        decisions[cell] = {
            "passed": passed,
            "success_rate": success_rate,
            "success_ci_lower": ci_lower,
            "force_violation_rate": summary["force_violation_rate"],
            "controller_saturation_rate": summary["mean_controller_saturation_rate"],
        }
    return {"passed": all(value["passed"] for value in decisions.values()), "cells": decisions}


def validate_lift_friction(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml("configs/evaluation/lift_friction_validation.yaml")
    if config.get("version") != VALIDATION_VERSION or config.get("task") != "lift":
        raise ValueError("Unsupported Lift friction-validation configuration")
    seeds, uniform_start, uniform_episodes, boundary_start, boundary_episodes = _resolve_protocol(
        config, args.mode
    )
    candidate_ood = _validate_candidate_ood(config)
    environment_hash = current_environment_hash("lift")
    refinement_sha256 = None
    if args.mode == "validation":
        refinement_sha256 = _validate_refinement_artifact(
            Path(config["source_refinement_artifact"]),
            environment_hash=environment_hash,
            refinement_config_hash=_json_hash(
                load_yaml("configs/evaluation/lift_friction_refinement.yaml")
            ),
        )

    feasibility_config = load_yaml("configs/evaluation/lift_feasibility.yaml")
    env_config = load_yaml("configs/env/lift.yaml")
    profiles = _validate_profiles(
        feasibility_config["controller_profiles"], env_config["robosuite"]["execution"]
    )
    profile_name = str(config["protocol"]["controller_profile"])
    if profile_name != "high_grip":
        raise ValueError("Lift support validation is locked to high_grip")
    profile = profiles[profile_name]
    manifest = _make_manifest(
        config=config,
        seeds=seeds,
        uniform_start=uniform_start,
        uniform_episodes=uniform_episodes,
        boundary_start=boundary_start,
        boundary_episodes=boundary_episodes,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    interval = max(1, len(manifest) // 20)
    for completed, spec in enumerate(manifest, start=1):
        try:
            rows.append(_run_episode(spec, profile_name, profile))
        except Exception as error:  # noqa: BLE001 - preserve validation attrition
            failures.append({**spec, "exception": f"{type(error).__name__}: {error}"})
        if completed % interval == 0 or completed == len(manifest):
            print(f"[{args.mode} lift friction validation] {completed}/{len(manifest)}", flush=True)

    complete = len(rows) == len(manifest) and not failures
    summaries = _summaries(rows) if complete else {}
    if complete and args.mode == "validation":
        gate = _gate_decision(summaries, config["gate"])
    else:
        gate = {"passed": complete, "scope": "plumbing_only"}
    result = {
        "validation_version": VALIDATION_VERSION,
        "mode": args.mode,
        "status": "PASS" if gate["passed"] else "FAIL",
        "scope": "held-out candidate Lift low-friction support validation",
        "task": "lift",
        "seeds": seeds,
        "uniform_episodes_per_seed": uniform_episodes,
        "boundary_episodes_per_seed": boundary_episodes,
        "environment_hash": environment_hash,
        "config_sha256": _json_hash(config),
        "candidate_ood_config_sha256": _json_hash(candidate_ood),
        "source_refinement_sha256": refinement_sha256,
        "expert_version": PrivilegedScriptedExpert.version,
        "controller_profile_name": profile_name,
        "controller_profile": profile,
        "thresholds": config["gate"],
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
        "summaries": summaries,
        "gate": gate,
        "failures": failures,
        "rows": rows,
        "warning": (
            "PASS validates only privileged feasibility of the candidate Lift support. The "
            "candidate is not active in learned-policy or oracle evaluation until separately frozen."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"manifest", "rows"}},
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and not gate["passed"]:
        raise RuntimeError("Held-out Lift friction-support validation failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    validate_lift_friction(build_parser().parse_args())


if __name__ == "__main__":
    main()
