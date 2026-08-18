"""Bounded privileged spatial-residual gate for the Lift adapter interface."""

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
from tdpa.controllers.spatial_residual_oracle import PrivilegedSpatialResidualOracle
from tdpa.data.nominal_demonstrations import file_sha256
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import physics_ood_path, validate_lift_physics_activation
from tdpa.evaluation.evaluate_nominal_policy import (
    _bootstrap_interval,
    _json_hash,
    _validate_competence_artifact,
)
from tdpa.evaluation.robosuite_oracle_v2 import (
    DEFAULT_CELLS,
    ORACLE_V2_GATE_VERSION,
    _make_manifest,
    _trace_update,
    _validate_execution_contract,
)
from tdpa.policies.learned_nominal import (
    FrozenLearnedNominalPolicy,
    checkpoint_sha256,
    current_environment_hash,
)
from tdpa.utils.config import load_yaml

SPATIAL_GATE_VERSION = "tdpa-robosuite-spatial-residual-gate-v1"
METHODS = ("b0", "privileged_spatial_residual")
CELLS = DEFAULT_CELLS["lift"]


def _validate_source_artifact(
    path: Path,
    *,
    checkpoint_hash: str,
    environment_hash: str,
    competence_hash: str,
    oracle_config_hash: str,
    adapter_config_hash: str,
    physics_ood_config_hash: str,
    lift_friction_validation_hash: str,
) -> tuple[dict[str, Any], str]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError("Spatial gate source artifact must be a mapping")
    expected = {
        "oracle_gate_version": ORACLE_V2_GATE_VERSION,
        "mode": "development",
        "status": "FAIL",
        "task": "lift",
        "checkpoint_sha256": checkpoint_hash,
        "environment_hash": environment_hash,
        "competence_artifact_sha256": competence_hash,
        "oracle_config_sha256": oracle_config_hash,
        "adapter_config_sha256": adapter_config_hash,
        "physics_ood_config_sha256": physics_ood_config_hash,
        "lift_friction_validation_sha256": lift_friction_validation_hash,
        "oracle_revision": 3,
        "cells": list(CELLS),
        "seeds": [4101, 4102, 4103],
        "episodes_per_seed_cell": 20,
        "reset_index_start": 40_000,
        "methods": ["b0", "perfect_context_oracle_v2"],
    }
    mismatched = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatched:
        raise ValueError(f"Spatial gate source artifact mismatch: {sorted(mismatched)}")
    if artifact.get("failures"):
        raise ValueError("Spatial gate source artifact has rollout attrition")
    if artifact.get("gate", {}).get("passed") is not False:
        raise ValueError("Spatial gate source must be the failed physics-only development gate")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, list) or len(manifest) != 180:
        raise ValueError("Spatial gate source manifest is incomplete")
    if artifact.get("manifest_sha256") != _json_hash(manifest):
        raise ValueError("Spatial gate source manifest hash is invalid")
    locked_manifest = _make_manifest(
        task="lift",
        cells=CELLS,
        seeds=[4101, 4102, 4103],
        reset_index_start=40_000,
        episodes=20,
    )
    if manifest != locked_manifest:
        raise ValueError("Spatial gate source does not use the current locked physics manifest")
    gate_cells = artifact.get("gate", {}).get("cells", {})
    if gate_cells.get("ood_mass_high", {}).get("recovery_pass") is not False:
        raise ValueError("Spatial gate requires the observed high-mass oracle failure")
    if not all(
        gate_cells.get(cell, {}).get("recovery_pass") is True
        for cell in ("ood_friction_low", "ood_composition")
    ):
        raise ValueError("Spatial gate source does not contain the two passing recovery cells")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise TypeError("Spatial gate source has no auditable rows")
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("Spatial gate source rows must be mappings")
    if not isinstance(artifact.get("summaries"), dict):
        raise TypeError("Spatial gate source has no summaries")
    try:
        nominal_success = float(artifact["gate"]["nominal_success_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Spatial gate source has no nominal competence summary") from error
    if not np.isfinite(nominal_success):
        raise ValueError("Spatial gate source nominal competence is non-finite")
    required_row_fields = {
        "success",
        "final_error",
        "force_violation",
        "saturation_rate",
        "reset_fingerprint",
    }
    if any(required_row_fields - set(row) for row in rows):
        raise ValueError("Spatial gate source rows are missing required diagnostics")
    b0_rows = [row for row in rows if row.get("method") == "b0"]
    physics_rows = [row for row in rows if row.get("method") == "perfect_context_oracle_v2"]
    if len(b0_rows) != len(physics_rows) or len(b0_rows) != len(manifest):
        raise ValueError("Spatial gate source method rows are incomplete")
    manifest_by_key = {
        (row["cell"], int(row["seed"]), int(row["episode"])): row for row in manifest
    }
    if len(manifest_by_key) != len(manifest):
        raise ValueError("Spatial gate source manifest keys are not unique")
    b0_by_key = {(row["cell"], int(row["seed"]), int(row["episode"])): row for row in b0_rows}
    physics_by_key = {
        (row["cell"], int(row["seed"]), int(row["episode"])): row for row in physics_rows
    }
    if set(b0_by_key) != set(manifest_by_key) or set(physics_by_key) != set(manifest_by_key):
        raise ValueError("Spatial gate source rows are not one-to-one with the manifest")
    for key, row in b0_by_key.items():
        spec = manifest_by_key.get(key)
        if spec is None or any(
            row[field] != spec[field] for field in ("reset_index", "mass", "friction")
        ):
            raise ValueError("Spatial gate B0 rows do not match the source manifest")
        physics_row = physics_by_key[key]
        if any(physics_row[field] != spec[field] for field in ("reset_index", "mass", "friction")):
            raise ValueError("Spatial gate physics rows do not match the source manifest")
        if row.get("reset_fingerprint") != physics_row.get("reset_fingerprint"):
            raise ValueError("Spatial gate source methods are not reset-paired")
    return artifact, file_sha256(path)


def _reject_checkpoint_overlap(
    policy: FrozenLearnedNominalPolicy, manifest: list[dict[str, Any]]
) -> None:
    trained_ids = set(policy.provenance.get("train_episode_ids", [])) | set(
        policy.provenance.get("validation_episode_ids", [])
    )
    collection_seed = policy.provenance.get("collection_seed")
    if any(
        int(spec["seed"]) == collection_seed and int(spec["reset_index"]) in trained_ids
        for spec in manifest
    ):
        raise ValueError("Spatial-gate reset manifest overlaps checkpoint training data")


def _run_episode(
    spec: dict[str, Any],
    *,
    method: str,
    policy: FrozenLearnedNominalPolicy,
    mapper: AdapterActionMapper,
    oracle: PrivilegedSpatialResidualOracle,
    expected_reset_fingerprint: str | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown spatial-gate method: {method}")
    physics = Physics(float(spec["mass"]), float(spec["friction"]))
    env = make_env(
        "lift",
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
    residual_norms: list[float] = []
    residual_abs_maxima: list[float] = []
    spatial_phases: Counter[str] = Counter()
    physics_phases: Counter[str] = Counter()
    projected_steps = 0
    near_bound_steps = 0
    gripper_disagreement_steps = 0
    saturation_steps = 0
    backend_saturation_steps = 0
    last_decision = None
    try:
        observation = env.reset()
        _validate_execution_contract(env, mapper)
        fingerprint = env.reset_fingerprint()
        if expected_reset_fingerprint is not None and fingerprint != expected_reset_fingerprint:
            raise RuntimeError("Spatial rollout reset does not match its source B0 row")
        reset_state = env.reset_state()
        readback = env.read_physics()
        if not np.isclose(readback.actual_mass, physics.mass, rtol=1e-9, atol=1e-12):
            raise RuntimeError("Live MuJoCo mass readback does not match the spatial manifest")
        friction_values = [
            geom.friction[0] for geom in (*readback.object_geoms, *readback.counterpart_geoms)
        ]
        if not friction_values or not np.allclose(
            friction_values, physics.friction, rtol=1e-9, atol=1e-12
        ):
            raise RuntimeError("Live MuJoCo friction readback does not match the spatial manifest")
        info = env.metrics()
        steps = 0
        for steps in range(1, env.horizon + 1):
            for key in ("rgbd", "proprio"):
                observation_digest.update(np.ascontiguousarray(observation[key]).tobytes())
            nominal_action = policy.act(observation)
            if method == "privileged_spatial_residual":
                last_decision = oracle.decide(env, physics, nominal_action, observation)
                applied = mapper.apply(nominal_action, last_decision.correction)
                action = applied.action
                controller = applied.controller
                if not np.isclose(
                    float(action[3]), float(np.clip(nominal_action[3], -1.0, 1.0)), atol=0.0
                ):
                    raise RuntimeError("Spatial oracle changed the nominal gripper command")
                projected_steps += int(last_decision.residual_projected)
                near_bound_steps += int(last_decision.residual_near_bound)
                gripper_disagreement_steps += int(last_decision.gripper_disagreement)
                residual_norms.append(
                    float(np.linalg.norm(last_decision.correction["cartesian_residual"]))
                )
                residual_abs_maxima.append(
                    float(np.max(np.abs(last_decision.correction["cartesian_residual"])))
                )
                spatial_phases[last_decision.expert_phase] += 1
                physics_phases[last_decision.physics_phase] += 1
                saturated = applied.saturated
            else:
                action = nominal_action
                controller = dict(mapper.nominal)
                saturated = False
            _trace_update(execution_digest, action, controller)
            observation, _, terminated, truncated, info = env.step(
                action,
                controller if method == "privileged_spatial_residual" else None,
            )
            backend_saturated = bool(info["controller_saturated"])
            backend_saturation_steps += int(backend_saturated)
            saturation_steps += int(saturated or backend_saturated)
            forces.append(float(info["contact_force"]))
            if terminated or truncated:
                break
        force_array = np.asarray(forces, dtype=np.float64)
        result: dict[str, Any] = {
            **spec,
            "method": method,
            "reset_fingerprint": fingerprint,
            "reset_state": reset_state.as_dict(),
            "physics_readback": readback.as_dict(),
            "success": bool(info["success"]),
            "final_error": float(info["final_error"]),
            "completion_time": float(info["completion_time"]),
            "steps": steps,
            "peak_contact_force": float(force_array.max(initial=0.0)),
            "rms_contact_force": float(np.sqrt(np.mean(np.square(force_array)))),
            "force_violation": bool(force_array.max(initial=0.0) > env.force_limit),
            "saturation_rate": saturation_steps / steps,
            "backend_saturation_rate": backend_saturation_steps / steps,
            "residual_projection_rate": projected_steps / steps,
            "residual_near_bound_rate": near_bound_steps / steps,
            "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
            "maximum_residual_abs": max(residual_abs_maxima, default=0.0),
            "gripper_disagreement_rate": gripper_disagreement_steps / steps,
            "nominal_gripper_override_rate": 0.0,
            "spatial_phase_counts": dict(sorted(spatial_phases.items())),
            "physics_phase_counts": dict(sorted(physics_phases.items())),
            "observation_trace_sha256": observation_digest.hexdigest(),
            "execution_trace_sha256": execution_digest.hexdigest(),
            "controller_readback": env.controller_readback(),
            "last_desired_residual": (
                last_decision.desired_residual.tolist() if last_decision is not None else None
            ),
            "last_bounded_residual": (
                np.asarray(last_decision.correction["cartesian_residual"]).tolist()
                if last_decision is not None
                else None
            ),
            "privileged_inputs": (
                ["mass", "friction", "object_pose", "target_pose", "grasp_state"]
                if method == "privileged_spatial_residual"
                else []
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


def _summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for cell in CELLS:
            selected = [row for row in rows if row["method"] == method and row["cell"] == cell]
            summaries[f"{method}/{cell}"] = {
                "episodes": len(selected),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "mean_final_error": float(np.mean([row["final_error"] for row in selected])),
                "force_violation_rate": float(
                    np.mean([row["force_violation"] for row in selected])
                ),
                "mean_saturation_rate": float(
                    np.mean([row.get("saturation_rate", 0.0) for row in selected])
                ),
                "mean_backend_saturation_rate": float(
                    np.mean([row.get("backend_saturation_rate", 0.0) for row in selected])
                ),
                "mean_residual_projection_rate": float(
                    np.mean([row.get("residual_projection_rate", 0.0) for row in selected])
                ),
                "mean_residual_near_bound_rate": float(
                    np.mean([row.get("residual_near_bound_rate", 0.0) for row in selected])
                ),
                "maximum_residual_abs": float(
                    max(row.get("maximum_residual_abs", 0.0) for row in selected)
                ),
                "mean_gripper_disagreement_rate": float(
                    np.mean([row.get("gripper_disagreement_rate", 0.0) for row in selected])
                ),
                "mean_nominal_gripper_override_rate": float(
                    np.mean([row.get("nominal_gripper_override_rate", 0.0) for row in selected])
                ),
            }
            for diagnostic in ("drop", "slip", "overshoot"):
                if selected and diagnostic in selected[0]:
                    summaries[f"{method}/{cell}"][f"{diagnostic}_rate"] = float(
                        np.mean([bool(row[diagnostic]) for row in selected])
                    )
    return summaries


def _paired_recovery(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
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
        result[cell] = {
            "pairs": len(keys),
            "reset_fingerprints_match": all(
                by_method[METHODS[0]][key]["reset_fingerprint"]
                == by_method[METHODS[1]][key]["reset_fingerprint"]
                for key in keys
            ),
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
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for cell in CELLS:
        b0 = summaries[f"b0/{cell}"]
        spatial = summaries[f"privileged_spatial_residual/{cell}"]
        absolute_recovery = float(recovery[cell]["success_recovery"])
        recovery_ci_lower = float(recovery[cell]["success_recovery_bootstrap_95_interval"][0])
        gap = max(nominal_success - float(b0["success_rate"]), 0.0)
        gap_closed = absolute_recovery / gap if gap > 0.0 else 0.0
        force_increase = float(spatial["force_violation_rate"]) - float(b0["force_violation_rate"])
        decisions[cell] = {
            "b0_success_rate": b0["success_rate"],
            "spatial_oracle_success_rate": spatial["success_rate"],
            "absolute_success_recovery": absolute_recovery,
            "success_recovery_ci_lower": recovery_ci_lower,
            "nominal_gap_closed": gap_closed,
            "force_violation_rate_increase": force_increase,
            "pairing_pass": bool(recovery[cell]["reset_fingerprints_match"]),
            "success_floor_pass": float(spatial["success_rate"])
            >= float(thresholds["minimum_oracle_success_rate"]),
            "recovery_pass": (
                absolute_recovery >= float(thresholds["minimum_absolute_success_recovery"])
                and recovery_ci_lower >= float(thresholds["minimum_success_recovery_ci_lower"])
                and gap_closed >= float(thresholds["minimum_gap_closed"])
            ),
            "bounded_execution_pass": float(spatial["mean_saturation_rate"])
            <= float(thresholds["maximum_saturation_rate"]),
            "residual_bound_pass": float(spatial["maximum_residual_abs"])
            <= float(thresholds["maximum_cartesian_residual_abs"]),
            "gripper_preservation_pass": float(spatial["mean_nominal_gripper_override_rate"])
            == 0.0,
            "force_diagnostic_pass": force_increase
            <= float(thresholds["maximum_force_violation_increase"]),
        }
    passed = all(
        all(value for key, value in decision.items() if key.endswith("_pass"))
        for decision in decisions.values()
    )
    return {"passed": passed, "nominal_success_rate": nominal_success, "cells": decisions}


def evaluate_spatial_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml("configs/oracle/robosuite_spatial_residual_v1.yaml")
    if config.get("version") != "tdpa-robosuite-spatial-residual-v1":
        raise ValueError("Unsupported spatial-residual gate configuration")
    if float(config["gate"]["maximum_cartesian_residual_abs"]) != float(
        config["correction"]["residual_max"]
    ):
        raise ValueError("Spatial-residual gate and execution bounds disagree")
    protocol = config["protocol"]
    if (
        protocol["seeds"] != [4101, 4102, 4103]
        or int(protocol["reset_index_start"]) != 40_000
        or int(protocol["episodes_per_seed_cell"]) != 20
        or tuple(protocol["cells"]) != CELLS
    ):
        raise ValueError("Spatial-residual development protocol is not locked")
    adapter_config = load_yaml("configs/adapter/lift.yaml")
    physics_oracle_config = load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    environment_hash = current_environment_hash("lift")
    adapter_config_hash = _json_hash(adapter_config)
    physics_oracle_config_hash = _json_hash(physics_oracle_config)
    physics_ood_config_hash = _json_hash(load_yaml(physics_ood_path("lift")))
    competence_hash: str | None = None
    validation_hash: str | None = None
    source_hash: str | None = None
    source_artifact: dict[str, Any] | None = None
    if args.mode == "development":
        if (
            args.competence_artifact is None
            or args.lift_friction_validation is None
            or args.source_oracle_development is None
        ):
            raise ValueError(
                "Development requires competence, friction-validation, and source-oracle artifacts"
            )
        competence_hash = _validate_competence_artifact(
            args.competence_artifact, task="lift", checkpoint_hash=checkpoint_hash
        )
        validation_hash = validate_lift_physics_activation(
            args.lift_friction_validation, environment_hash=environment_hash
        )
        source_artifact, source_hash = _validate_source_artifact(
            args.source_oracle_development,
            checkpoint_hash=checkpoint_hash,
            environment_hash=environment_hash,
            competence_hash=competence_hash,
            oracle_config_hash=physics_oracle_config_hash,
            adapter_config_hash=adapter_config_hash,
            physics_ood_config_hash=physics_ood_config_hash,
            lift_friction_validation_hash=validation_hash,
        )
        manifest = source_artifact["manifest"]
        source_b0_rows = [row for row in source_artifact["rows"] if row["method"] == "b0"]
        nominal_success = float(source_artifact["gate"]["nominal_success_rate"])
    else:
        if any(
            value is not None
            for value in (
                args.competence_artifact,
                args.lift_friction_validation,
                args.source_oracle_development,
            )
        ):
            raise ValueError("Smoke mode must not consume development artifacts")
        manifest = _make_manifest(
            task="lift",
            cells=CELLS,
            seeds=[11],
            reset_index_start=60_000,
            episodes=1,
        )
        source_b0_rows = []
        nominal_success = 0.0

    policy = FrozenLearnedNominalPolicy(
        args.checkpoint,
        task="lift",
        device=args.device,
        allow_untrained=args.mode == "smoke",
    )
    _reject_checkpoint_overlap(policy, manifest)
    mapper = AdapterActionMapper(adapter_config)
    oracle = PrivilegedSpatialResidualOracle(
        config,
        physics_oracle_config,
        adapter_config,
        position_delta_limit=float(
            load_yaml("configs/env/lift.yaml")["robosuite"]["position_delta_limit"]
        ),
        enabled=True,
    )
    b0_by_key = {
        (row["cell"], int(row["seed"]), int(row["episode"])): row for row in source_b0_rows
    }
    rows: list[dict[str, Any]] = list(source_b0_rows)
    failures: list[dict[str, Any]] = []
    methods = METHODS if args.mode == "smoke" else ("privileged_spatial_residual",)
    total = len(manifest) * len(methods)
    interval = max(1, total // 20)
    completed = 0
    for spec in manifest:
        for method in methods:
            key = (spec["cell"], int(spec["seed"]), int(spec["episode"]))
            expected_fingerprint = b0_by_key[key]["reset_fingerprint"] if key in b0_by_key else None
            try:
                rows.append(
                    _run_episode(
                        spec,
                        method=method,
                        policy=policy,
                        mapper=mapper,
                        oracle=oracle,
                        expected_reset_fingerprint=expected_fingerprint,
                    )
                )
            except Exception as error:  # noqa: BLE001 - preserve manifest attrition
                failures.append(
                    {**spec, "method": method, "exception": f"{type(error).__name__}: {error}"}
                )
            completed += 1
            if completed % interval == 0 or completed == total:
                print(f"[{args.mode} Lift spatial residual] {completed}/{total}", flush=True)

    expected_rows = len(manifest) * len(METHODS)
    complete = len(rows) == expected_rows and not failures
    summaries = _summaries(rows) if complete else {}
    recovery = _paired_recovery(rows) if complete else {}
    if complete and args.mode == "development":
        gate = _gate_decision(
            nominal_success=nominal_success,
            summaries=summaries,
            recovery=recovery,
            thresholds=config["gate"],
        )
    else:
        gate = {"passed": complete, "scope": "plumbing_only"}
    result = {
        "spatial_gate_version": SPATIAL_GATE_VERSION,
        "spatial_oracle_version": PrivilegedSpatialResidualOracle.version,
        "spatial_oracle_revision": config["revision"],
        "mode": args.mode,
        "status": "PASS" if gate["passed"] else "FAIL",
        "scope": "privileged bounded spatial-residual adapter-interface upper bound",
        "task": "lift",
        "cells": list(CELLS),
        "seeds": protocol["seeds"] if args.mode == "development" else [11],
        "episodes_per_seed_cell": (
            int(protocol["episodes_per_seed_cell"]) if args.mode == "development" else 1
        ),
        "reset_index_start": (
            int(protocol["reset_index_start"]) if args.mode == "development" else 60_000
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "environment_hash": environment_hash,
        "competence_artifact_sha256": competence_hash,
        "lift_friction_validation_sha256": validation_hash,
        "source_oracle_development_sha256": source_hash,
        "spatial_oracle_config_sha256": _json_hash(config),
        "physics_oracle_config_sha256": physics_oracle_config_hash,
        "adapter_config_sha256": adapter_config_hash,
        "physics_ood_config_sha256": physics_ood_config_hash,
        "manifest": manifest,
        "manifest_sha256": _json_hash(manifest),
        "methods": list(METHODS),
        "gate_thresholds": config["gate"],
        "gate": gate,
        "summaries": summaries,
        "paired_recovery": recovery,
        "source_physics_oracle_summaries": (
            source_artifact["summaries"] if source_artifact is not None else None
        ),
        "failures": failures,
        "rows": rows,
        "warning": (
            "The spatial oracle consumes privileged object pose, target pose, and grasp state in "
            "addition to mass/friction. It preserves the nominal gripper command and the deployed "
            "adapter/controller bounds. PASS is an interface upper bound, not deployable adaptation."
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
        raise RuntimeError("Privileged spatial-residual development gate failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competence-artifact", type=Path)
    parser.add_argument("--lift-friction-validation", type=Path)
    parser.add_argument("--source-oracle-development", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    evaluate_spatial_gate(build_parser().parse_args())


if __name__ == "__main__":
    main()
