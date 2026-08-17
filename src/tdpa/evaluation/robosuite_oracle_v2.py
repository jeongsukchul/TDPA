"""Leakage-separated development/final gate for the robosuite oracle v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.controllers.oracle_context import (
    OracleV2Decision,
    RobosuitePerfectContextOracleV2,
)
from tdpa.data.nominal_demonstrations import file_sha256
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import physics_ood_path, validate_lift_physics_activation
from tdpa.evaluation.evaluate_nominal_policy import (
    _bootstrap_interval,
    _json_hash,
    _physics_for_cell,
    _validate_competence_artifact,
)
from tdpa.policies.learned_nominal import (
    FrozenLearnedNominalPolicy,
    checkpoint_sha256,
    current_environment_hash,
)
from tdpa.utils.config import load_yaml

ORACLE_V2_GATE_VERSION = "tdpa-robosuite-oracle-gate-v2"
METHODS = ("b0", "perfect_context_oracle_v2")
DEFAULT_CELLS = {
    "push": ("ood_friction_high",),
    "lift": ("ood_mass_high", "ood_friction_low", "ood_composition"),
}


def _validate_execution_contract(env: Any, mapper: AdapterActionMapper) -> None:
    execution = env.robosuite_config.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("Live robosuite environment has no execution contract")
    nominal = execution.get("nominal")
    bounds = execution.get("bounds")
    if not isinstance(nominal, dict) or not isinstance(bounds, dict):
        raise TypeError("Live execution contract must contain nominal values and bounds")
    for key in ("velocity_scale", "stiffness", "damping", "grip_force"):
        if not np.isclose(float(nominal[key]), float(mapper.nominal[key]), atol=0.0, rtol=0.0):
            raise ValueError(f"Adapter nominal {key} does not match the live environment")
        if not np.array_equal(
            np.asarray(bounds[key], dtype=np.float64),
            np.asarray(mapper.bounds[key], dtype=np.float64),
        ):
            raise ValueError(f"Adapter bounds for {key} do not match the live environment")


def _stage_protocol(
    config: dict[str, Any], mode: str
) -> tuple[list[int], int, int]:
    if mode == "smoke":
        return [11], 60_000, 1
    stage = config["protocol"][mode]
    seeds = [int(seed) for seed in stage["seeds"]]
    episodes = int(config["protocol"]["episodes_per_seed_cell"])
    reset_index_start = int(stage["reset_index_start"])
    if len(seeds) != 3 or len(set(seeds)) != 3 or episodes != 20:
        raise ValueError("Oracle-v2 development/final protocols are locked to 3x20")
    if mode == "development" and set(seeds) != {4101, 4102, 4103}:
        raise ValueError("Oracle-v2 development seeds are locked")
    if mode == "final" and set(seeds) != {5101, 5102, 5103}:
        raise ValueError("Oracle-v2 final seeds are locked")
    return seeds, reset_index_start, episodes


def _make_manifest(
    *,
    task: str,
    cells: tuple[str, ...],
    seeds: list[int],
    reset_index_start: int,
    episodes: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for cell in cells:
        for seed in seeds:
            for episode in range(episodes):
                physics = _physics_for_cell(cell, seed=seed, episode=episode, task=task)
                manifest.append(
                    {
                        "task": task,
                        "cell": cell,
                        "seed": seed,
                        "episode": episode,
                        "reset_index": reset_index_start + episode,
                        "mass": physics.mass,
                        "friction": physics.friction,
                    }
                )
    return manifest


def _load_competence(
    path: Path | None, *, task: str, checkpoint_hash: str, mode: str
) -> tuple[float | None, str | None]:
    if mode == "smoke":
        if path is not None:
            raise ValueError("Smoke mode must not consume a competence result")
        return None, None
    if path is None:
        raise ValueError("--competence-artifact is required outside smoke mode")
    digest = _validate_competence_artifact(path, task=task, checkpoint_hash=checkpoint_hash)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    try:
        nominal_success = float(artifact["summaries"]["nominal"]["success_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Competence artifact has no nominal success summary") from error
    return nominal_success, digest


def _validate_development_artifact(
    path: Path | None,
    *,
    task: str,
    checkpoint_hash: str,
    environment_hash: str,
    oracle_config_hash: str,
    adapter_config_hash: str,
    physics_ood_config_hash: str | None = None,
    lift_friction_validation_hash: str | None = None,
) -> str:
    if path is None:
        raise ValueError("--development-artifact is required in final mode")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "oracle_gate_version": ORACLE_V2_GATE_VERSION,
        "mode": "development",
        "status": "PASS",
        "task": task,
        "checkpoint_sha256": checkpoint_hash,
        "environment_hash": environment_hash,
        "oracle_config_sha256": oracle_config_hash,
        "adapter_config_sha256": adapter_config_hash,
    }
    if task == "lift":
        if physics_ood_config_hash is None or lift_friction_validation_hash is None:
            raise ValueError("Lift final gate requires calibrated-friction provenance")
        expected.update(
            {
                "physics_ood_config_sha256": physics_ood_config_hash,
                "lift_friction_validation_sha256": lift_friction_validation_hash,
            }
        )
    elif artifact.get("physics_ood_config_sha256") is not None:
        expected["physics_ood_config_sha256"] = physics_ood_config_hash
    mismatched = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatched:
        raise ValueError(
            f"Final gate requires a matching passing development artifact: {sorted(mismatched)}"
        )
    if artifact.get("seeds") != [4101, 4102, 4103]:
        raise ValueError("Development artifact does not use the locked seeds")
    if artifact.get("cells") != list(DEFAULT_CELLS[task]):
        raise ValueError("Development artifact does not use the locked hard cells")
    if artifact.get("episodes_per_seed_cell") != 20 or artifact.get("failures"):
        raise ValueError("Development artifact is incomplete")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, list) or artifact.get("manifest_sha256") != _json_hash(manifest):
        raise ValueError("Development artifact manifest is invalid")
    return file_sha256(path)


def _json_safe_command(command: dict[str, Any] | None) -> dict[str, Any] | None:
    if command is None:
        return None
    result: dict[str, Any] = {}
    for key, value in command.items():
        array = np.asarray(value)
        result[key] = (
            float(array.reshape(-1)[0]) if array.size == 1 else array.astype(float).tolist()
        )
    return result


def _command_projected(decision: OracleV2Decision) -> bool:
    for key in ("velocity_scale", "stiffness", "damping", "grip_force"):
        if not np.isclose(
            float(decision.raw_correction[key]),
            float(decision.correction[key]),
            rtol=0.0,
            atol=0.0,
        ):
            return True
    return not np.array_equal(
        np.asarray(decision.raw_correction["cartesian_residual"]),
        np.asarray(decision.correction["cartesian_residual"]),
    )


def _trace_update(digest: Any, action: np.ndarray, controller: dict[str, float]) -> None:
    values = np.concatenate(
        [
            np.asarray(action, dtype=np.float32),
            np.asarray(
                [
                    controller["velocity_scale"],
                    controller["stiffness"],
                    controller["damping"],
                    controller["grip_force"],
                ],
                dtype=np.float32,
            ),
        ]
    )
    digest.update(np.ascontiguousarray(values).tobytes())


def _run_episode(
    spec: dict[str, Any],
    *,
    method: str,
    policy: FrozenLearnedNominalPolicy,
    mapper: AdapterActionMapper,
    oracle: RobosuitePerfectContextOracleV2,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown oracle-v2 method: {method}")
    task = str(spec["task"])
    physics = Physics(float(spec["mass"]), float(spec["friction"]))
    env = make_env(
        task,
        physics=physics,
        seed=int(spec["seed"]),
        episode_index=int(spec["reset_index"]),
        backend="robosuite",
    )
    policy.reset()
    oracle.reset()
    observation_digest = hashlib.sha256()
    execution_digest = hashlib.sha256()
    forces: list[float] = []
    phases: Counter[str] = Counter()
    action_projection_steps = 0
    schedule_projection_steps = 0
    backend_saturation_steps = 0
    saturation_steps = 0
    last_decision: OracleV2Decision | None = None
    try:
        observation = env.reset()
        _validate_execution_contract(env, mapper)
        fingerprint = env.reset_fingerprint()
        reset_state = env.reset_state()
        readback = env.read_physics()
        if not np.isclose(readback.actual_mass, physics.mass, rtol=1e-9, atol=1e-12):
            raise RuntimeError("Live MuJoCo mass readback does not match the manifest")
        friction_values = [
            geom.friction[0] for geom in (*readback.object_geoms, *readback.counterpart_geoms)
        ]
        if not friction_values or not np.allclose(
            friction_values, physics.friction, rtol=1e-9, atol=1e-12
        ):
            raise RuntimeError("Live MuJoCo friction readback does not match the manifest")
        info = env.metrics()
        steps = 0
        for steps in range(1, env.horizon + 1):
            for key in ("rgbd", "proprio"):
                observation_digest.update(np.ascontiguousarray(observation[key]).tobytes())
            nominal_action = policy.act(observation)
            if method == "perfect_context_oracle_v2":
                last_decision = oracle.decide(physics, nominal_action, observation)
                applied = mapper.apply(nominal_action, last_decision.correction)
                requested_motion = (
                    nominal_action[:3] * float(last_decision.correction["velocity_scale"])
                    + np.asarray(last_decision.correction["cartesian_residual"])
                )
                action_projected = bool(np.any(np.abs(requested_motion) > 1.0))
                schedule_projected = _command_projected(last_decision)
                action = applied.action
                controller = applied.controller
                phases[last_decision.phase] += 1
            else:
                action = nominal_action
                controller = dict(mapper.nominal)
                action_projected = False
                schedule_projected = False
                phases["b0"] += 1
            _trace_update(execution_digest, action, controller)
            observation, _, terminated, truncated, info = env.step(
                action,
                controller if method == "perfect_context_oracle_v2" else None,
            )
            backend_saturated = bool(info["controller_saturated"])
            action_projection_steps += int(action_projected)
            schedule_projection_steps += int(schedule_projected)
            backend_saturation_steps += int(backend_saturated)
            saturation_steps += int(
                action_projected or backend_saturated or (method != "b0" and applied.saturated)
            )
            forces.append(float(info["contact_force"]))
            if terminated or truncated:
                break
        force_array = np.asarray(forces, dtype=np.float64)
        result: dict[str, Any] = {
            **spec,
            "reset_fingerprint": fingerprint,
            "reset_state": reset_state.as_dict(),
            "physics_readback": readback.as_dict(),
            "method": method,
            "success": bool(info["success"]),
            "final_error": float(info["final_error"]),
            "completion_time": float(info["completion_time"]),
            "steps": steps,
            "peak_contact_force": float(force_array.max(initial=0.0)),
            "rms_contact_force": float(np.sqrt(np.mean(np.square(force_array)))),
            "force_violation": bool(force_array.max(initial=0.0) > env.force_limit),
            "action_projection_rate": action_projection_steps / steps,
            "schedule_projection_rate": schedule_projection_steps / steps,
            "backend_saturation_rate": backend_saturation_steps / steps,
            "saturation_rate": saturation_steps / steps,
            "phase_counts": dict(phases),
            "observation_trace_sha256": observation_digest.hexdigest(),
            "execution_trace_sha256": execution_digest.hexdigest(),
            "controller_readback": env.controller_readback(),
            "privileged_inputs": ["mass", "friction"] if method != "b0" else [],
            "last_raw_oracle_command": _json_safe_command(
                last_decision.raw_correction if last_decision is not None else None
            ),
            "last_bounded_oracle_command": _json_safe_command(
                last_decision.correction if last_decision is not None else None
            ),
            "renderer": os.environ.get("MUJOCO_GL", "default"),
            "versions": env.versions(),
            "exception": None,
        }
        for key in ("overshoot", "drop", "slip"):
            if key in info:
                result[key] = info[key]
        return result
    finally:
        env.close()


def _summaries(rows: list[dict[str, Any]], cells: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for cell in cells:
            selected = [row for row in rows if row["method"] == method and row["cell"] == cell]
            summary = {
                "episodes": len(selected),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "mean_final_error": float(np.mean([row["final_error"] for row in selected])),
                "force_violation_rate": float(
                    np.mean([row["force_violation"] for row in selected])
                ),
                "mean_action_projection_rate": float(
                    np.mean([row["action_projection_rate"] for row in selected])
                ),
                "mean_schedule_projection_rate": float(
                    np.mean([row["schedule_projection_rate"] for row in selected])
                ),
                "mean_backend_saturation_rate": float(
                    np.mean([row["backend_saturation_rate"] for row in selected])
                ),
                "mean_saturation_rate": float(
                    np.mean([row["saturation_rate"] for row in selected])
                ),
            }
            for diagnostic in ("drop", "slip", "overshoot"):
                if selected and diagnostic in selected[0]:
                    summary[f"{diagnostic}_rate"] = float(
                        np.mean([bool(row[diagnostic]) for row in selected])
                    )
            summaries[f"{method}/{cell}"] = summary
    return summaries


def _paired_recovery(
    rows: list[dict[str, Any]], cells: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cell in cells:
        by_method = {
            method: {
                (int(row["seed"]), int(row["episode"])): row
                for row in rows
                if row["method"] == method and row["cell"] == cell
            }
            for method in METHODS
        }
        keys = sorted(set(by_method[METHODS[0]]) & set(by_method[METHODS[1]]))
        success = np.asarray(
            [
                float(by_method[METHODS[1]][key]["success"])
                - float(by_method[METHODS[0]][key]["success"])
                for key in keys
            ],
            dtype=np.float64,
        )
        error = np.asarray(
            [
                float(by_method[METHODS[0]][key]["final_error"])
                - float(by_method[METHODS[1]][key]["final_error"])
                for key in keys
            ],
            dtype=np.float64,
        )
        paired_fingerprints = all(
            by_method[METHODS[0]][key]["reset_fingerprint"]
            == by_method[METHODS[1]][key]["reset_fingerprint"]
            for key in keys
        )
        result[cell] = {
            "pairs": len(keys),
            "reset_fingerprints_match": paired_fingerprints,
            "success_recovery": float(success.mean()),
            "success_recovery_bootstrap_95_interval": _bootstrap_interval(success),
            "mean_final_error_reduction": float(error.mean()),
            "final_error_reduction_bootstrap_95_interval": _bootstrap_interval(error),
        }
    return result


def _gate_decision(
    *,
    nominal_success: float,
    summaries: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
    cells: tuple[str, ...],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for cell in cells:
        b0 = summaries[f"b0/{cell}"]
        oracle = summaries[f"perfect_context_oracle_v2/{cell}"]
        absolute_recovery = float(recovery[cell]["success_recovery"])
        recovery_ci_lower = float(recovery[cell]["success_recovery_bootstrap_95_interval"][0])
        gap = max(nominal_success - float(b0["success_rate"]), 0.0)
        gap_closed = absolute_recovery / gap if gap > 0 else 0.0
        force_increase = float(oracle["force_violation_rate"]) - float(
            b0["force_violation_rate"]
        )
        decisions[cell] = {
            "b0_success_rate": b0["success_rate"],
            "oracle_success_rate": oracle["success_rate"],
            "absolute_success_recovery": absolute_recovery,
            "success_recovery_ci_lower": recovery_ci_lower,
            "nominal_gap_closed": gap_closed,
            "force_violation_rate_increase": force_increase,
            "pairing_pass": bool(recovery[cell]["reset_fingerprints_match"]),
            "recovery_pass": (
                absolute_recovery
                >= float(gate_config["minimum_absolute_success_recovery"])
                and recovery_ci_lower
                >= float(gate_config["minimum_success_recovery_ci_lower"])
                and gap_closed >= float(gate_config["minimum_gap_closed"])
            ),
            "bounded_execution_pass": (
                float(oracle["mean_saturation_rate"])
                <= float(gate_config["maximum_saturation_rate"])
            ),
            "force_diagnostic_pass": (
                force_increase <= float(gate_config["maximum_force_violation_increase"])
            ),
        }
    passed = all(all(value for key, value in decision.items() if key.endswith("_pass")) for decision in decisions.values())
    return {"passed": passed, "nominal_success_rate": nominal_success, "cells": decisions}


def evaluate_oracle_v2(args: argparse.Namespace) -> dict[str, Any]:
    oracle_config = load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")
    adapter_config = load_yaml(f"configs/adapter/{args.task}.yaml")
    cells = DEFAULT_CELLS[args.task]
    seeds, reset_index_start, episodes = _stage_protocol(oracle_config, args.mode)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    environment_hash = current_environment_hash(args.task)
    oracle_config_hash = _json_hash(oracle_config)
    adapter_config_hash = _json_hash(adapter_config)
    physics_ood_config_hash = _json_hash(load_yaml(physics_ood_path(args.task)))
    lift_friction_validation_hash: str | None = None
    if args.task == "lift" and args.mode != "smoke":
        if args.lift_friction_validation is None:
            raise ValueError("Lift oracle-v2 requires --lift-friction-validation")
        lift_friction_validation_hash = validate_lift_physics_activation(
            args.lift_friction_validation,
            environment_hash=environment_hash,
        )
    elif args.lift_friction_validation is not None:
        raise ValueError("--lift-friction-validation is accepted only for non-smoke Lift oracle-v2")
    nominal_success, competence_hash = _load_competence(
        args.competence_artifact,
        task=args.task,
        checkpoint_hash=checkpoint_hash,
        mode=args.mode,
    )
    development_hash: str | None = None
    if args.mode == "final":
        development_hash = _validate_development_artifact(
            args.development_artifact,
            task=args.task,
            checkpoint_hash=checkpoint_hash,
            environment_hash=environment_hash,
            oracle_config_hash=oracle_config_hash,
            adapter_config_hash=adapter_config_hash,
            physics_ood_config_hash=physics_ood_config_hash,
            lift_friction_validation_hash=lift_friction_validation_hash,
        )
    elif args.development_artifact is not None:
        raise ValueError("--development-artifact is accepted only in final mode")

    manifest = _make_manifest(
        task=args.task,
        cells=cells,
        seeds=seeds,
        reset_index_start=reset_index_start,
        episodes=episodes,
    )
    policy = FrozenLearnedNominalPolicy(
        args.checkpoint,
        task=args.task,
        device=args.device,
        allow_untrained=args.mode == "smoke",
    )
    trained_ids = set(policy.provenance.get("train_episode_ids", [])) | set(
        policy.provenance.get("validation_episode_ids", [])
    )
    collection_seed = policy.provenance.get("collection_seed")
    if any(
        int(spec["seed"]) == collection_seed and int(spec["reset_index"]) in trained_ids
        for spec in manifest
    ):
        raise ValueError("Oracle-v2 reset manifest overlaps checkpoint training data")

    mapper = AdapterActionMapper(adapter_config)
    oracle = RobosuitePerfectContextOracleV2(
        args.task, oracle_config, adapter_config, enabled=True
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_rollouts = len(manifest) * len(METHODS)
    progress_interval = max(1, total_rollouts // 20)
    completed_rollouts = 0
    for spec in manifest:
        for method in METHODS:
            try:
                rows.append(
                    _run_episode(
                        spec,
                        method=method,
                        policy=policy,
                        mapper=mapper,
                        oracle=oracle,
                    )
                )
            except Exception as error:  # noqa: BLE001 - preserve manifest attrition
                failures.append(
                    {
                        **spec,
                        "method": method,
                        "exception": f"{type(error).__name__}: {error}",
                    }
                )
            completed_rollouts += 1
            if completed_rollouts % progress_interval == 0 or completed_rollouts == total_rollouts:
                print(
                    f"[{args.mode} {args.task}] {completed_rollouts}/{total_rollouts} rollouts",
                    flush=True,
                )

    complete = len(rows) == len(manifest) * len(METHODS) and not failures
    summaries = _summaries(rows, cells) if complete else {}
    recovery = _paired_recovery(rows, cells) if complete else {}
    if complete and args.mode != "smoke":
        if nominal_success is None:
            raise RuntimeError("Non-smoke oracle-v2 gate lost its competence summary")
        gate = _gate_decision(
            nominal_success=nominal_success,
            summaries=summaries,
            recovery=recovery,
            cells=cells,
            gate_config=oracle_config["gate"],
        )
    else:
        gate = {"passed": complete, "scope": "plumbing_only"}
    result = {
        "oracle_gate_version": ORACLE_V2_GATE_VERSION,
        "mode": args.mode,
        "status": "PASS" if gate["passed"] else "FAIL",
        "scope": "bounded phase-preserving perfect-context controller upper bound",
        "task": args.task,
        "cells": list(cells),
        "seeds": seeds,
        "episodes_per_seed_cell": episodes,
        "reset_index_start": reset_index_start,
        "environment_hash": environment_hash,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "competence_artifact_sha256": competence_hash,
        "development_artifact_sha256": development_hash,
        "oracle_config_sha256": oracle_config_hash,
        "adapter_config_sha256": adapter_config_hash,
        "physics_ood_config_sha256": physics_ood_config_hash,
        "lift_friction_validation_sha256": lift_friction_validation_hash,
        "oracle_revision": oracle_config["revision"],
        "oracle_schedule": oracle_config[args.task],
        "gate_thresholds": oracle_config["gate"],
        "manifest_sha256": _json_hash(manifest),
        "manifest": manifest,
        "methods": list(METHODS),
        "summaries": summaries,
        "paired_recovery": recovery,
        "gate": gate,
        "failures": failures,
        "rows": rows,
        "warning": (
            "Only mass/friction are privileged. Development results may tune later revisions; "
            "final mode is valid only for the identical schedule frozen by its passing development "
            "artifact. Lift low-friction support requires its held-out feasibility PASS. MuJoCo "
            "force values remain uncalibrated diagnostics."
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
        raise RuntimeError(f"Robosuite oracle-v2 {args.mode} gate failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development", "final"), required=True)
    parser.add_argument("--task", choices=("push", "lift"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competence-artifact", type=Path)
    parser.add_argument("--development-artifact", type=Path)
    parser.add_argument("--lift-friction-validation", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    evaluate_oracle_v2(build_parser().parse_args())


if __name__ == "__main__":
    main()
