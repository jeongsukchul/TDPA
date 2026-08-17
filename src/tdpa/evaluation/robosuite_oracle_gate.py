"""Paired perfect-context upper-bound gate for learned robosuite policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.controllers.oracle_context import RobosuitePerfectContextOracle
from tdpa.data.nominal_demonstrations import file_sha256
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.evaluation.evaluate_nominal_policy import (
    _bootstrap_interval,
    _json_hash,
    _validate_competence_artifact,
)
from tdpa.policies.learned_nominal import (
    FrozenLearnedNominalPolicy,
    checkpoint_sha256,
    current_environment_hash,
)
from tdpa.utils.config import load_yaml

ORACLE_GATE_VERSION = "tdpa-robosuite-oracle-gate-v1"
METHODS = ("b0", "perfect_context_oracle")
DEFAULT_CELLS = {
    "push": ("ood_friction_high",),
    "lift": ("ood_mass_high", "ood_friction_low", "ood_composition"),
}


def _validate_source_artifacts(
    *,
    mode: str,
    task: str,
    checkpoint: Path,
    competence_artifact: Path | None,
    ood_artifact: Path,
) -> tuple[dict[str, Any], str | None]:
    checkpoint_hash = checkpoint_sha256(checkpoint)
    source = json.loads(ood_artifact.read_text(encoding="utf-8"))
    allowed_modes = {"smoke"} if mode == "smoke" else {"ood"}
    if source.get("status") != "PASS" or source.get("mode") not in allowed_modes:
        raise ValueError(f"Oracle {mode} requires a passing {sorted(allowed_modes)} artifact")
    if source.get("task") != task or source.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("OOD artifact task/checkpoint does not match oracle evaluation")
    if source.get("environment_hash") != current_environment_hash(task):
        raise ValueError("OOD artifact environment does not match the current task configuration")
    source_manifest = _json_hash(
        [
            {key: row[key] for key in ("task", "cell", "seed", "episode", "mass", "friction")}
            for row in source.get("rows", [])
        ]
        + source.get("failures", [])
    )
    if source.get("manifest_sha256") != source_manifest:
        raise ValueError("OOD artifact manifest hash is invalid")
    competence_hash: str | None = None
    if mode == "full":
        if competence_artifact is None:
            raise ValueError("--competence-artifact is required in full mode")
        competence_hash = _validate_competence_artifact(
            competence_artifact,
            task=task,
            checkpoint_hash=checkpoint_hash,
        )
        if source.get("competence_artifact_sha256") != competence_hash:
            raise ValueError("OOD artifact does not reference the supplied competence artifact")
        if len(source.get("seeds", [])) != 3 or source.get("episodes_per_seed_cell") != 20:
            raise ValueError("Full oracle gate requires the locked 3x20 OOD artifact")
    return source, competence_hash


def _source_rows(
    source: dict[str, Any], *, cells: tuple[str, ...], mode: str
) -> list[dict[str, Any]]:
    available = set(source.get("cells", []))
    missing = set(cells) - available
    if missing:
        raise ValueError(f"OOD artifact is missing requested cells: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for cell in cells:
        selected = sorted(
            (row for row in source["rows"] if row["cell"] == cell),
            key=lambda row: (int(row["seed"]), int(row["episode"])),
        )
        if mode == "smoke":
            selected = selected[:1]
        expected = 1 if mode == "smoke" else 60
        if len(selected) != expected:
            raise ValueError(f"Expected {expected} source rows for {cell}, got {len(selected)}")
        keys = {(int(row["seed"]), int(row["episode"])) for row in selected}
        if len(keys) != len(selected):
            raise ValueError(f"Duplicate source manifest rows for {cell}")
        if mode == "full":
            expected_keys = {
                (int(seed), episode) for seed in source["seeds"] for episode in range(20)
            }
            if keys != expected_keys:
                raise ValueError(f"Source rows for {cell} do not match the locked seed/index grid")
        rows.extend(selected)
    return rows


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
        live_interval = np.asarray(bounds[key], dtype=np.float64)
        adapter_interval = np.asarray(mapper.bounds[key], dtype=np.float64)
        if not np.array_equal(live_interval, adapter_interval):
            raise ValueError(f"Adapter bounds for {key} do not match the live environment")


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
    source_row: dict[str, Any],
    *,
    task: str,
    method: str,
    policy: FrozenLearnedNominalPolicy,
    mapper: AdapterActionMapper,
    oracle: RobosuitePerfectContextOracle,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown oracle method: {method}")
    physics = Physics(float(source_row["mass"]), float(source_row["friction"]))
    env = make_env(
        task,
        physics=physics,
        seed=int(source_row["seed"]),
        episode_index=int(source_row["reset_index"]),
        backend="robosuite",
    )
    policy.reset()
    observation_digest = hashlib.sha256()
    execution_digest = hashlib.sha256()
    forces: list[float] = []
    saturated_steps = 0
    raw_oracle_command: dict[str, Any] | None = None
    bounded_oracle_command: dict[str, Any] | None = None
    try:
        observation = env.reset()
        _validate_execution_contract(env, mapper)
        fingerprint = env.reset_fingerprint()
        if fingerprint != source_row["reset_fingerprint"]:
            raise RuntimeError("Oracle rollout reset does not match its source OOD row")
        readback = env.read_physics()
        if not np.isclose(readback.actual_mass, physics.mass, rtol=1e-9, atol=1e-12):
            raise RuntimeError("Live MuJoCo mass readback does not match the source manifest")
        friction_values = [
            geom.friction[0] for geom in (*readback.object_geoms, *readback.counterpart_geoms)
        ]
        if not friction_values or not np.allclose(
            friction_values, physics.friction, rtol=1e-9, atol=1e-12
        ):
            raise RuntimeError("Live MuJoCo friction readback does not match the source manifest")
        info = env.metrics()
        steps = 0
        for steps in range(1, env.horizon + 1):
            for key in ("rgbd", "proprio"):
                observation_digest.update(np.ascontiguousarray(observation[key]).tobytes())
            nominal_action = policy.act(observation)
            if method == "perfect_context_oracle":
                raw_oracle_command = oracle.raw_correction(physics)
                bounded_oracle_command = oracle.correction(physics)
                applied = mapper.apply(nominal_action, bounded_oracle_command)
                action = applied.action
                controller = applied.controller
                saturated = applied.saturated
            else:
                action = nominal_action
                controller = dict(mapper.nominal)
                saturated = False
            _trace_update(execution_digest, action, controller)
            observation, _, terminated, truncated, info = env.step(
                action,
                controller if method == "perfect_context_oracle" else None,
            )
            forces.append(float(info["contact_force"]))
            saturated_steps += int(saturated or bool(info["controller_saturated"]))
            if terminated or truncated:
                break
        force_array = np.asarray(forces, dtype=np.float64)
        result: dict[str, Any] = {
            "task": task,
            "cell": source_row["cell"],
            "seed": int(source_row["seed"]),
            "episode": int(source_row["episode"]),
            "reset_index": int(source_row["reset_index"]),
            "reset_fingerprint": fingerprint,
            "mass": physics.mass,
            "friction": physics.friction,
            "physics_readback": readback.as_dict(),
            "method": method,
            "success": bool(info["success"]),
            "final_error": float(info["final_error"]),
            "completion_time": float(info["completion_time"]),
            "steps": steps,
            "peak_contact_force": float(force_array.max(initial=0.0)),
            "rms_contact_force": float(np.sqrt(np.mean(np.square(force_array)))),
            "force_violation": bool(force_array.max(initial=0.0) > env.force_limit),
            "saturation_rate": saturated_steps / steps,
            "observation_trace_sha256": observation_digest.hexdigest(),
            "execution_trace_sha256": execution_digest.hexdigest(),
            "controller_readback": env.controller_readback(),
            "privileged_physics_used": method == "perfect_context_oracle",
            "raw_oracle_command": (
                _json_safe_command(raw_oracle_command) if raw_oracle_command is not None else None
            ),
            "bounded_oracle_command": (
                _json_safe_command(bounded_oracle_command)
                if bounded_oracle_command is not None
                else None
            ),
            "source_b0_success": bool(source_row["success"]),
            "source_b0_final_error": float(source_row["final_error"]),
            "source_b0_success_reproduced": (
                method != "b0" or bool(info["success"]) == bool(source_row["success"])
            ),
            "source_b0_final_error_delta": (
                None
                if method != "b0"
                else float(info["final_error"]) - float(source_row["final_error"])
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


def _json_safe_command(command: dict[str, Any]) -> dict[str, float | list[float]]:
    result: dict[str, float | list[float]] = {}
    for key, value in command.items():
        array = np.asarray(value)
        result[key] = float(array.reshape(-1)[0]) if array.size == 1 else array.astype(float).tolist()
    return result


def _summaries(rows: list[dict[str, Any]], cells: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for cell in cells:
            selected = [row for row in rows if row["method"] == method and row["cell"] == cell]
            result[f"{method}/{cell}"] = {
                "episodes": len(selected),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "mean_final_error": float(np.mean([row["final_error"] for row in selected])),
                "force_violation_rate": float(
                    np.mean([row["force_violation"] for row in selected])
                ),
                "mean_saturation_rate": float(
                    np.mean([row["saturation_rate"] for row in selected])
                ),
            }
    return result


def _paired_recovery(
    rows: list[dict[str, Any]], cells: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cell in cells:
        by_method = {
            method: {
                (row["seed"], row["episode"]): row
                for row in rows
                if row["method"] == method and row["cell"] == cell
            }
            for method in METHODS
        }
        keys = sorted(set(by_method["b0"]) & set(by_method["perfect_context_oracle"]))
        success = np.asarray(
            [
                float(by_method["perfect_context_oracle"][key]["success"])
                - float(by_method["b0"][key]["success"])
                for key in keys
            ],
            dtype=np.float64,
        )
        error = np.asarray(
            [
                float(by_method["b0"][key]["final_error"])
                - float(by_method["perfect_context_oracle"][key]["final_error"])
                for key in keys
            ],
            dtype=np.float64,
        )
        result[cell] = {
            "pairs": len(keys),
            "success_recovery": float(success.mean()),
            "success_recovery_bootstrap_95_interval": _bootstrap_interval(success),
            "mean_final_error_reduction": float(error.mean()),
            "final_error_reduction_bootstrap_95_interval": _bootstrap_interval(error),
        }
    return result


def _gate_decision(
    *,
    source: dict[str, Any],
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
    cells: tuple[str, ...],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    nominal_success = float(source["summaries"]["nominal"]["success_rate"])
    decisions: dict[str, Any] = {}
    for cell in cells:
        b0 = summaries[f"b0/{cell}"]
        oracle = summaries[f"perfect_context_oracle/{cell}"]
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
    replay_rows = [row for row in rows if row["method"] == "b0"]
    b0_reproduced = bool(replay_rows) and all(
        row["source_b0_success_reproduced"] for row in replay_rows
    )
    passed = b0_reproduced and all(
        decision["recovery_pass"]
        and decision["bounded_execution_pass"]
        and decision["force_diagnostic_pass"]
        for decision in decisions.values()
    )
    return {
        "passed": passed,
        "b0_success_reproduced": b0_reproduced,
        "nominal_success_rate": nominal_success,
        "cells": decisions,
    }


def evaluate_oracle_gate(args: argparse.Namespace) -> dict[str, Any]:
    cells = tuple(args.cells or DEFAULT_CELLS[args.task])
    if args.mode == "full" and cells != DEFAULT_CELLS[args.task]:
        raise ValueError(f"Full {args.task} oracle gate cells are locked to {DEFAULT_CELLS[args.task]}")
    source, competence_hash = _validate_source_artifacts(
        mode=args.mode,
        task=args.task,
        checkpoint=args.checkpoint,
        competence_artifact=args.competence_artifact,
        ood_artifact=args.ood_artifact,
    )
    source_rows = _source_rows(source, cells=cells, mode=args.mode)
    adapter_config = load_yaml(f"configs/adapter/{args.task}.yaml")
    oracle_config = load_yaml("configs/oracle/robosuite_perfect_context.yaml")
    mapper = AdapterActionMapper(adapter_config)
    oracle = RobosuitePerfectContextOracle(
        args.task,
        oracle_config,
        adapter_config,
        enabled=True,
    )
    policy = FrozenLearnedNominalPolicy(
        args.checkpoint,
        task=args.task,
        device=args.device,
        allow_untrained=args.mode == "smoke",
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_row in source_rows:
        for method in METHODS:
            try:
                rows.append(
                    _run_episode(
                        source_row,
                        task=args.task,
                        method=method,
                        policy=policy,
                        mapper=mapper,
                        oracle=oracle,
                    )
                )
            except Exception as error:  # noqa: BLE001 - record manifest attrition
                failures.append(
                    {
                        "task": args.task,
                        "cell": source_row["cell"],
                        "seed": source_row["seed"],
                        "episode": source_row["episode"],
                        "method": method,
                        "exception": f"{type(error).__name__}: {error}",
                    }
                )
    complete = len(rows) == len(source_rows) * len(METHODS) and not failures
    summaries = _summaries(rows, cells) if complete else {}
    recovery = _paired_recovery(rows, cells) if complete else {}
    gate = (
        _gate_decision(
            source=source,
            rows=rows,
            summaries=summaries,
            recovery=recovery,
            cells=cells,
            gate_config=oracle_config["gate"],
        )
        if complete and args.mode == "full"
        else {"passed": complete, "scope": "smoke_only"}
    )
    result = {
        "oracle_gate_version": ORACLE_GATE_VERSION,
        "mode": args.mode,
        "status": "PASS" if gate["passed"] else "FAIL",
        "scope": "bounded perfect-context controller upper bound on frozen learned policy",
        "task": args.task,
        "cells": list(cells),
        "environment_hash": current_environment_hash(args.task),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "competence_artifact_sha256": competence_hash,
        "ood_artifact": str(args.ood_artifact),
        "ood_artifact_sha256": file_sha256(args.ood_artifact),
        "source_manifest_sha256": source["manifest_sha256"],
        "oracle_config_sha256": _json_hash(oracle_config),
        "oracle_schedule": {
            "nominal_physics": oracle_config["nominal_physics"],
            "task_schedule": oracle_config[args.task],
        },
        "gate_thresholds": oracle_config["gate"],
        "adapter_config_sha256": _json_hash(adapter_config),
        "methods": list(METHODS),
        "summaries": summaries,
        "paired_recovery": recovery,
        "gate": gate,
        "failures": failures,
        "rows": rows,
        "warning": (
            "The oracle sees true mass/friction and is not deployable. MuJoCo force diagnostics "
            "are uncalibrated; this result cannot establish representation-learning claims."
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
    if args.strict and not gate["passed"]:
        raise RuntimeError("Robosuite perfect-context oracle gate failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--task", choices=("push", "lift"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competence-artifact", type=Path)
    parser.add_argument("--ood-artifact", type=Path, required=True)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    evaluate_oracle_gate(build_parser().parse_args())


if __name__ == "__main__":
    main()
