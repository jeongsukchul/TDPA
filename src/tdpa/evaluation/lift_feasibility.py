"""No-training privileged-expert feasibility audit for extreme Lift physics."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.envs.base import Physics, PhysicsReadback
from tdpa.envs.make_env import make_env
from tdpa.evaluation.evaluate_nominal_policy import (
    _bootstrap_interval,
    _json_hash,
    _physics_for_cell,
)
from tdpa.policies.learned_nominal import current_environment_hash
from tdpa.policies.privileged_expert import PrivilegedScriptedExpert
from tdpa.utils.config import load_yaml

FEASIBILITY_VERSION = "tdpa-lift-feasibility-v1"


def _resolve_protocol(
    config: dict[str, Any], mode: str
) -> tuple[tuple[str, ...], list[int], int, int]:
    cells = tuple(str(cell) for cell in config["protocol"]["cells"])
    if cells != ("ood_mass_high", "ood_friction_low"):
        raise ValueError("Lift feasibility cells are locked to high mass and low friction")
    if mode == "smoke":
        return cells, [17], 80_000, 1
    seeds = [int(seed) for seed in config["protocol"]["seeds"]]
    episodes = int(config["protocol"]["episodes_per_seed_cell"])
    index_start = int(config["protocol"]["reset_index_start"])
    if seeds != [6101, 6102, 6103] or episodes != 20 or index_start != 70_000:
        raise ValueError("Lift feasibility development protocol is not the locked 3x20 manifest")
    return cells, seeds, index_start, episodes


def _make_manifest(
    *, cells: tuple[str, ...], seeds: list[int], index_start: int, episodes: int
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for cell in cells:
        for seed in seeds:
            for episode in range(episodes):
                physics = _physics_for_cell(cell, seed=seed, episode=episode)
                manifest.append(
                    {
                        "task": "lift",
                        "cell": cell,
                        "seed": seed,
                        "episode": episode,
                        "reset_index": index_start + episode,
                        "mass": physics.mass,
                        "friction": physics.friction,
                    }
                )
    return manifest


def _validate_profiles(
    profiles: dict[str, Any], execution: dict[str, Any]
) -> dict[str, dict[str, float]]:
    controller_keys = {"velocity_scale", "stiffness", "damping", "grip_force"}
    optional_keys = {"lift_velocity_scale", "transport_velocity_scale"}
    bounds = execution["bounds"]
    validated: dict[str, dict[str, float]] = {}
    if set(profiles) != {"nominal", "high_grip", "high_authority", "gentle_lift"}:
        raise ValueError("Lift feasibility requires the four locked controller profiles")
    for name, raw_profile in profiles.items():
        keys = set(raw_profile)
        if not controller_keys <= keys or keys - controller_keys - optional_keys:
            raise ValueError(f"Controller profile {name} has an invalid schema")
        if name == "gentle_lift" and not optional_keys <= keys:
            raise ValueError("gentle_lift requires phase-specific velocity scales")
        if name != "gentle_lift" and keys & optional_keys:
            raise ValueError(f"Only gentle_lift may define phase-specific velocity for {name}")
        profile = {key: float(value) for key, value in raw_profile.items()}
        for key in controller_keys:
            value = profile[key]
            low, high = map(float, bounds[key])
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Controller profile {name}/{key} exceeds execution bounds")
        velocity_low, velocity_high = map(float, bounds["velocity_scale"])
        for key in optional_keys & keys:
            if not velocity_low <= profile[key] <= velocity_high:
                raise ValueError(f"Controller profile {name}/{key} exceeds execution bounds")
        validated[name] = profile
    return validated


def _controller_for_phase(profile: dict[str, float], phase: str) -> dict[str, float]:
    controller = {
        key: profile[key] for key in ("velocity_scale", "stiffness", "damping", "grip_force")
    }
    phase_key = f"{phase}_velocity_scale"
    if phase_key in profile:
        controller["velocity_scale"] = profile[phase_key]
    return controller


def _grasped(env: Any) -> bool:
    return bool(
        env.raw._check_grasp(
            gripper=env.raw.robots[0].gripper,
            object_geoms=env.raw.cube,
        )
    )


def _validate_physics_readback(readback: PhysicsReadback, physics: Physics) -> None:
    if readback.backend != "robosuite":
        raise RuntimeError("Lift feasibility requires a live robosuite/MuJoCo readback")
    if not np.isclose(readback.actual_mass, physics.mass, rtol=1e-9, atol=1e-12):
        raise RuntimeError("Live MuJoCo mass readback does not match the feasibility manifest")
    friction_values = [
        geom.friction[0] for geom in (*readback.object_geoms, *readback.counterpart_geoms)
    ]
    if not friction_values or not np.allclose(
        friction_values, physics.friction, rtol=1e-9, atol=1e-12
    ):
        raise RuntimeError("Live MuJoCo friction readback does not match the feasibility manifest")


def _failure_stage(
    *,
    success: bool,
    contact_steps: int,
    grasped_steps: int,
    final_grasped: bool,
    grasp_losses: int,
) -> str | None:
    if success:
        return None
    if contact_steps == 0:
        return "no_fingerpad_contact"
    if grasped_steps == 0:
        return "contact_without_grasp"
    if grasp_losses > 0 or not final_grasped:
        return "grasp_lost"
    return "lift_or_transport_timeout"


def _run_episode(
    spec: dict[str, Any], profile_name: str, profile: dict[str, float]
) -> dict[str, Any]:
    physics = Physics(float(spec["mass"]), float(spec["friction"]))
    env = make_env(
        "lift",
        physics=physics,
        seed=int(spec["seed"]),
        episode_index=int(spec["reset_index"]),
        backend="robosuite",
    )
    expert = PrivilegedScriptedExpert(
        "lift",
        position_delta_limit=float(env.robosuite_config["position_delta_limit"]),
    )
    phase_counts: Counter[str] = Counter()
    relevant_contact_steps = 0
    both_pad_contact_steps = 0
    grasped_steps = 0
    saturation_steps = 0
    forces: list[float] = []
    first_contact_step: int | None = None
    first_grasp_step: int | None = None
    try:
        profiles = _validate_profiles(
            {**load_yaml("configs/evaluation/lift_feasibility.yaml")["controller_profiles"]},
            env.robosuite_config["execution"],
        )
        if profiles[profile_name] != profile:
            raise RuntimeError("Feasibility controller profile changed after manifest construction")
        observation = env.reset()
        fingerprint = env.reset_fingerprint()
        reset_state = env.reset_state()
        readback = env.read_physics()
        _validate_physics_readback(readback, physics)
        initial_object_position = np.asarray(
            env.raw.sim.data.body_xpos[env._body_id], dtype=np.float64
        ).copy()
        target_position = np.asarray(
            env.raw.sim.data.site_xpos[env._target_site_id], dtype=np.float64
        ).copy()
        initial_object_height = float(initial_object_position[2])
        maximum_object_height = initial_object_height
        info = env.metrics()
        steps = 0
        for steps in range(1, env.horizon + 1):
            decision = expert.act(env, observation)
            phase_counts[decision.phase] += 1
            controller = _controller_for_phase(profile, decision.phase)
            observation, _, terminated, truncated, info = env.step(decision.action, controller)
            report = env.contact_report()
            contact = int(report["relevant_contact_count"]) > 0
            pad_counts = list(report["counterpart_contact_counts"].values())
            both_pads = len(pad_counts) == 2 and all(int(count) > 0 for count in pad_counts)
            currently_grasped = _grasped(env)
            relevant_contact_steps += int(contact)
            both_pad_contact_steps += int(both_pads)
            grasped_steps += int(currently_grasped)
            saturation_steps += int(bool(info["controller_saturated"]))
            force = float(info["contact_force"])
            forces.append(force)
            if contact and first_contact_step is None:
                first_contact_step = steps
            if currently_grasped and first_grasp_step is None:
                first_grasp_step = steps
            maximum_object_height = max(
                maximum_object_height,
                float(env.raw.sim.data.body_xpos[env._body_id][2]),
            )
            if terminated or truncated:
                break
        force_array = np.asarray(forces, dtype=np.float64)
        success = bool(info["success"])
        final_grasped = _grasped(env)
        final_object_position = np.asarray(
            env.raw.sim.data.body_xpos[env._body_id], dtype=np.float64
        ).copy()
        failure_stage = _failure_stage(
            success=success,
            contact_steps=relevant_contact_steps,
            grasped_steps=grasped_steps,
            final_grasped=final_grasped,
            grasp_losses=expert.grasp_losses,
        )
        return {
            **spec,
            "profile": profile_name,
            "controller_schedule": profile,
            "reset_fingerprint": fingerprint,
            "reset_state": reset_state.as_dict(),
            "physics_readback": readback.as_dict(),
            "success": success,
            "final_error": float(info["final_error"]),
            "completion_time": float(info["completion_time"]),
            "steps": steps,
            "phase_counts": dict(sorted(phase_counts.items())),
            "expert_phase_index": expert.phase,
            "first_contact_step": first_contact_step,
            "first_grasp_step": first_grasp_step,
            "relevant_contact_rate": relevant_contact_steps / steps,
            "both_pad_contact_rate": both_pad_contact_steps / steps,
            "grasped_rate": grasped_steps / steps,
            "ever_contacted": relevant_contact_steps > 0,
            "ever_grasped": grasped_steps > 0,
            "final_grasped": final_grasped,
            "expert_grasp_losses": expert.grasp_losses,
            "initial_object_position": initial_object_position.tolist(),
            "final_object_position": final_object_position.tolist(),
            "target_position": target_position.tolist(),
            "initial_object_height": initial_object_height,
            "maximum_object_height": maximum_object_height,
            "maximum_lift_height": maximum_object_height - initial_object_height,
            "peak_contact_force": float(force_array.max(initial=0.0)),
            "force_violation": bool(force_array.max(initial=0.0) > env.force_limit),
            "controller_saturation_rate": saturation_steps / steps,
            "controller_readback": env.controller_readback(),
            "failure_stage": failure_stage,
            "renderer": os.environ.get("MUJOCO_GL", "default"),
            "versions": env.versions(),
            "exception": None,
        }
    finally:
        env.close()


def _summaries(
    rows: list[dict[str, Any]],
    *,
    cells: tuple[str, ...],
    profiles: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cell in cells:
        for profile in profiles:
            selected = [row for row in rows if row["cell"] == cell and row["profile"] == profile]
            successes = np.asarray([float(row["success"]) for row in selected], dtype=np.float64)
            failure_stages = Counter(
                str(row["failure_stage"]) for row in selected if row["failure_stage"] is not None
            )
            result[f"{profile}/{cell}"] = {
                "episodes": len(selected),
                "success_rate": float(successes.mean()),
                "success_bootstrap_95_interval": _bootstrap_interval(successes),
                "mean_final_error": float(np.mean([row["final_error"] for row in selected])),
                "ever_contacted_rate": float(np.mean([row["ever_contacted"] for row in selected])),
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
                "total_grasp_losses": int(sum(row["expert_grasp_losses"] for row in selected)),
                "failure_stages": dict(sorted(failure_stages.items())),
            }
    return result


def _gate_decision(
    summaries: dict[str, dict[str, Any]],
    *,
    cells: tuple[str, ...],
    profiles: dict[str, dict[str, float]],
    thresholds: dict[str, Any],
    paired_resets_pass: bool,
) -> dict[str, Any]:
    cells_result: dict[str, Any] = {}
    for cell in cells:
        candidates: dict[str, Any] = {}
        for profile in profiles:
            summary = summaries[f"{profile}/{cell}"]
            success_rate = float(summary["success_rate"])
            ci_lower = float(summary["success_bootstrap_95_interval"][0])
            profile_pass = (
                success_rate >= float(thresholds["minimum_success_rate"])
                and ci_lower >= float(thresholds["minimum_success_ci_lower"])
                and float(summary["force_violation_rate"])
                <= float(thresholds["maximum_force_violation_rate"])
                and float(summary["mean_controller_saturation_rate"])
                <= float(thresholds["maximum_controller_saturation_rate"])
            )
            candidates[profile] = {
                "success_rate": success_rate,
                "success_ci_lower": ci_lower,
                "force_violation_rate": summary["force_violation_rate"],
                "controller_saturation_rate": summary["mean_controller_saturation_rate"],
                "passed": profile_pass,
            }
        best_profile = max(
            profiles,
            key=lambda name: (
                candidates[name]["success_rate"],
                -candidates[name]["force_violation_rate"],
            ),
        )
        passing_profiles = [name for name, value in candidates.items() if value["passed"]]
        cells_result[cell] = {
            "feasible": bool(passing_profiles),
            "best_profile": best_profile,
            "passing_profiles": passing_profiles,
            "profiles": candidates,
        }
    passed = paired_resets_pass and all(value["feasible"] for value in cells_result.values())
    return {
        "passed": passed,
        "paired_resets_pass": paired_resets_pass,
        "cells": cells_result,
    }


def evaluate_lift_feasibility(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml("configs/evaluation/lift_feasibility.yaml")
    if config.get("version") != FEASIBILITY_VERSION or config.get("task") != "lift":
        raise ValueError("Unsupported Lift feasibility configuration")
    cells, seeds, index_start, episodes = _resolve_protocol(config, args.mode)
    env_config = load_yaml("configs/env/lift.yaml")
    profiles = _validate_profiles(
        config["controller_profiles"], env_config["robosuite"]["execution"]
    )
    manifest = _make_manifest(
        cells=cells,
        seeds=seeds,
        index_start=index_start,
        episodes=episodes,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(manifest) * len(profiles)
    interval = max(1, total // 20)
    completed = 0
    for spec in manifest:
        for profile_name, profile in profiles.items():
            try:
                rows.append(_run_episode(spec, profile_name, profile))
            except Exception as error:  # noqa: BLE001 - preserve manifest attrition
                failures.append(
                    {
                        **spec,
                        "profile": profile_name,
                        "exception": f"{type(error).__name__}: {error}",
                    }
                )
            completed += 1
            if completed % interval == 0 or completed == total:
                print(f"[{args.mode} lift feasibility] {completed}/{total} rollouts", flush=True)

    complete = len(rows) == total and not failures
    fingerprints: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["episode"]))
        fingerprints.setdefault(key, set()).add(str(row["reset_fingerprint"]))
    paired_resets_pass = bool(fingerprints) and all(
        len(values) == 1 for values in fingerprints.values()
    )
    summaries = _summaries(rows, cells=cells, profiles=profiles) if complete else {}
    if complete and args.mode == "development":
        gate = _gate_decision(
            summaries,
            cells=cells,
            profiles=profiles,
            thresholds=config["gate"],
            paired_resets_pass=paired_resets_pass,
        )
    else:
        gate = {"passed": complete and paired_resets_pass, "scope": "plumbing_only"}
    result = {
        "feasibility_version": FEASIBILITY_VERSION,
        "mode": args.mode,
        "status": "PASS" if gate["passed"] else "FAIL",
        "scope": "privileged spatial-expert feasibility under the deployment controller envelope",
        "task": "lift",
        "cells": list(cells),
        "seeds": seeds,
        "episodes_per_seed_cell": episodes,
        "reset_index_start": index_start,
        "environment_hash": current_environment_hash("lift"),
        "config_sha256": _json_hash(config),
        "expert_version": PrivilegedScriptedExpert.version,
        "controller_profiles": profiles,
        "thresholds": config["gate"],
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
        "summaries": summaries,
        "gate": gate,
        "failures": failures,
        "rows": rows,
        "warning": (
            "The expert consumes privileged object/target pose and grasp state. PASS establishes "
            "simulator/control-envelope feasibility only; it is not a deployable-policy result or "
            "a safety claim."
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
        raise RuntimeError("Lift feasibility gate failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    evaluate_lift_feasibility(build_parser().parse_args())


if __name__ == "__main__":
    main()
