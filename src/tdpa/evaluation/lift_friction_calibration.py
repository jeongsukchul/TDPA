"""No-training empirical mass-friction calibration for the Lift grasp envelope."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.evaluation.evaluate_nominal_policy import _bootstrap_interval, _json_hash
from tdpa.evaluation.lift_feasibility import _run_episode, _validate_profiles
from tdpa.policies.learned_nominal import current_environment_hash
from tdpa.policies.privileged_expert import PrivilegedScriptedExpert
from tdpa.utils.config import load_yaml

CALIBRATION_VERSION = "tdpa-lift-friction-calibration-v1"


def _resolve_protocol(
    config: dict[str, Any], mode: str
) -> tuple[list[float], list[float], list[int], int, int]:
    protocol = config["protocol"]
    masses = [float(value) for value in protocol["mass_grid"]]
    frictions = [float(value) for value in protocol["friction_grid"]]
    if masses != [0.60, 0.90, 1.20, 1.40]:
        raise ValueError("Lift calibration mass grid is not locked")
    if frictions != [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
        raise ValueError("Lift calibration friction grid is not locked")
    if mode == "smoke":
        return [masses[0], masses[-1]], [frictions[0], frictions[-1]], [19], 95_000, 1
    seeds = [int(seed) for seed in protocol["seeds"]]
    episodes = int(protocol["episodes_per_seed_point"])
    index_start = int(protocol["reset_index_start"])
    if seeds != [7101, 7102, 7103] or episodes != 5 or index_start != 90_000:
        raise ValueError("Lift calibration development protocol is not locked")
    return masses, frictions, seeds, index_start, episodes


def _make_manifest(
    *,
    masses: list[float],
    frictions: list[float],
    seeds: list[int],
    index_start: int,
    episodes: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for mass_index, mass in enumerate(masses):
        for friction_index, friction in enumerate(frictions):
            for seed in seeds:
                for episode in range(episodes):
                    manifest.append(
                        {
                            "task": "lift",
                            "cell": "friction_calibration",
                            "mass_index": mass_index,
                            "friction_index": friction_index,
                            "seed": seed,
                            "episode": episode,
                            "reset_index": index_start + episode,
                            "mass": mass,
                            "friction": friction,
                        }
                    )
    return manifest


def _point_summaries(
    rows: list[dict[str, Any]], *, masses: list[float], frictions: list[float]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for mass in masses:
        for friction in frictions:
            selected = [
                row
                for row in rows
                if float(row["mass"]) == mass and float(row["friction"]) == friction
            ]
            successes = np.asarray([float(row["success"]) for row in selected], dtype=np.float64)
            failure_stages = Counter(
                str(row["failure_stage"]) for row in selected if row["failure_stage"] is not None
            )
            summaries.append(
                {
                    "mass": mass,
                    "friction": friction,
                    "episodes": len(selected),
                    "success_rate": float(successes.mean()),
                    "success_bootstrap_95_interval": _bootstrap_interval(successes),
                    "mean_final_error": float(np.mean([row["final_error"] for row in selected])),
                    "ever_grasped_rate": float(np.mean([row["ever_grasped"] for row in selected])),
                    "mean_grasped_rate": float(np.mean([row["grasped_rate"] for row in selected])),
                    "mean_maximum_lift_height": float(
                        np.mean([row["maximum_lift_height"] for row in selected])
                    ),
                    "force_violation_rate": float(
                        np.mean([row["force_violation"] for row in selected])
                    ),
                    "mean_controller_saturation_rate": float(
                        np.mean([row["controller_saturation_rate"] for row in selected])
                    ),
                    "failure_stages": dict(sorted(failure_stages.items())),
                }
            )
    return summaries


def _point_pass(summary: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return bool(
        float(summary["success_rate"]) >= float(thresholds["minimum_point_success_rate"])
        and float(summary["success_bootstrap_95_interval"][0])
        >= float(thresholds["minimum_point_success_ci_lower"])
        and float(summary["force_violation_rate"])
        <= float(thresholds["maximum_force_violation_rate"])
        and float(summary["mean_controller_saturation_rate"])
        <= float(thresholds["maximum_controller_saturation_rate"])
    )


def _frontier_decision(
    summaries: list[dict[str, Any]],
    *,
    masses: list[float],
    frictions: list[float],
    thresholds: dict[str, Any],
    paired_resets_pass: bool,
) -> dict[str, Any]:
    levels: list[dict[str, Any]] = []
    for friction in frictions:
        points = [summary for summary in summaries if float(summary["friction"]) == friction]
        mass_results = {
            f"{float(point['mass']):.2f}": {
                "success_rate": point["success_rate"],
                "success_ci_lower": point["success_bootstrap_95_interval"][0],
                "force_violation_rate": point["force_violation_rate"],
                "controller_saturation_rate": point["mean_controller_saturation_rate"],
                "passed": _point_pass(point, thresholds),
            }
            for point in points
        }
        levels.append(
            {
                "friction": friction,
                "all_masses_feasible": len(points) == len(masses)
                and all(value["passed"] for value in mass_results.values()),
                "worst_mass_success_rate": min(
                    float(value["success_rate"]) for value in mass_results.values()
                ),
                "mass_results": mass_results,
            }
        )

    minimum_levels = int(thresholds["minimum_contiguous_feasible_levels"])
    frontier_index: int | None = None
    for index in range(len(levels)):
        suffix = levels[index:]
        if len(suffix) >= minimum_levels and all(level["all_masses_feasible"] for level in suffix):
            frontier_index = index
            break
    recommended_support = None
    if frontier_index is not None:
        recommended_support = [levels[frontier_index]["friction"], levels[-1]["friction"]]
    passed = paired_resets_pass and recommended_support is not None
    return {
        "passed": passed,
        "paired_resets_pass": paired_resets_pass,
        "levels": levels,
        "recommended_low_friction_support": recommended_support,
        "recommendation_scope": "empirical grid only; not automatically applied",
    }


def calibrate_lift_friction(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml("configs/evaluation/lift_friction_calibration.yaml")
    if config.get("version") != CALIBRATION_VERSION or config.get("task") != "lift":
        raise ValueError("Unsupported Lift friction-calibration configuration")
    masses, frictions, seeds, index_start, episodes = _resolve_protocol(config, args.mode)
    feasibility_config = load_yaml("configs/evaluation/lift_feasibility.yaml")
    env_config = load_yaml("configs/env/lift.yaml")
    profiles = _validate_profiles(
        feasibility_config["controller_profiles"], env_config["robosuite"]["execution"]
    )
    profile_name = str(config["protocol"]["controller_profile"])
    if profile_name != "high_grip":
        raise ValueError("Lift friction calibration is locked to the high_grip profile")
    profile = profiles[profile_name]
    if profile["grip_force"] != float(
        env_config["robosuite"]["execution"]["bounds"]["grip_force"][1]
    ):
        raise ValueError("Calibration profile does not use the locked maximum grip-force bound")
    if max(frictions) >= float(config["gate"]["train_friction_lower_bound"]):
        raise ValueError("Calibration grid overlaps the declared training-friction support")

    manifest = _make_manifest(
        masses=masses,
        frictions=frictions,
        seeds=seeds,
        index_start=index_start,
        episodes=episodes,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    interval = max(1, len(manifest) // 20)
    for completed, spec in enumerate(manifest, start=1):
        try:
            rows.append(_run_episode(spec, profile_name, profile))
        except Exception as error:  # noqa: BLE001 - retain calibration attrition
            failures.append({**spec, "exception": f"{type(error).__name__}: {error}"})
        if completed % interval == 0 or completed == len(manifest):
            print(
                f"[{args.mode} lift friction calibration] {completed}/{len(manifest)}", flush=True
            )

    complete = len(rows) == len(manifest) and not failures
    fingerprints: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["episode"]))
        fingerprints.setdefault(key, set()).add(str(row["reset_fingerprint"]))
    paired_resets_pass = bool(fingerprints) and all(
        len(values) == 1 for values in fingerprints.values()
    )
    summaries = _point_summaries(rows, masses=masses, frictions=frictions) if complete else []
    if complete and args.mode == "development":
        gate = _frontier_decision(
            summaries,
            masses=masses,
            frictions=frictions,
            thresholds=config["gate"],
            paired_resets_pass=paired_resets_pass,
        )
    else:
        gate = {"passed": complete and paired_resets_pass, "scope": "plumbing_only"}

    result = {
        "calibration_version": CALIBRATION_VERSION,
        "mode": args.mode,
        "status": "PASS" if gate["passed"] else "FAIL",
        "scope": "privileged empirical Lift grasp-feasibility calibration",
        "task": "lift",
        "mass_grid": masses,
        "friction_grid": frictions,
        "seeds": seeds,
        "episodes_per_seed_point": episodes,
        "reset_index_start": index_start,
        "environment_hash": current_environment_hash("lift"),
        "config_sha256": _json_hash(config),
        "feasibility_config_sha256": _json_hash(feasibility_config),
        "expert_version": PrivilegedScriptedExpert.version,
        "controller_profile_name": profile_name,
        "controller_profile": profile,
        "thresholds": config["gate"],
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
        "point_summaries": summaries,
        "gate": gate,
        "failures": failures,
        "rows": rows,
        "warning": (
            "This privileged diagnostic calibrates only the current Panda / cube / OSC / gripper "
            "simulation envelope. It neither trains a policy nor changes the OOD distribution."
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
        raise RuntimeError("Lift friction calibration did not find a feasible contiguous support")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    calibrate_lift_friction(build_parser().parse_args())


if __name__ == "__main__":
    main()
