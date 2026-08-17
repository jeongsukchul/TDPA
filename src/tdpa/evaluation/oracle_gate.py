from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.controllers.oracle_context import OracleContextAdapter
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import load_physics_config
from tdpa.envs.physics_randomization import PhysicsRandomizer, PhysicsSplit
from tdpa.evaluation.metrics import aggregate_episode_metrics, force_summary
from tdpa.policies.behavior import apply_behavior_style, execution_trace_hash
from tdpa.policies.frozen_nominal import FrozenNominalPolicy
from tdpa.utils.checkpoints import git_commit, runtime_versions
from tdpa.utils.config import load_yaml


@dataclass(frozen=True)
class EpisodeManifest:
    task: str
    split: str
    seed: int
    index: int
    mass: float
    friction: float
    behavior_policy_id: str


def make_manifest(task: str, seeds: Iterable[int], episodes_per_split: int) -> list[EpisodeManifest]:
    physics_config = load_physics_config()
    manifest: list[EpisodeManifest] = []
    for seed in seeds:
        randomizer = PhysicsRandomizer(physics_config, seed)
        for split in PhysicsSplit:
            for index in range(episodes_per_split):
                sample = randomizer.sample_at(split, index)
                manifest.append(
                    EpisodeManifest(
                        task=task,
                        split=split.value,
                        seed=seed,
                        index=index,
                        mass=sample.mass,
                        friction=sample.friction,
                        behavior_policy_id=sample.behavior_policy_id,
                    )
                )
        # The nominal point is a benchmark competence check, not a randomization split.
        nominal = load_yaml("configs/physics/train.yaml")["nominal"]
        for index in range(episodes_per_split):
            manifest.append(
                EpisodeManifest(
                    task=task,
                    split="nominal",
                    seed=seed,
                    index=index,
                    mass=float(nominal["mass"]),
                    friction=float(nominal["friction"]),
                    behavior_policy_id="nominal",
                )
            )
    return manifest


def manifest_hash(manifest: Iterable[EpisodeManifest]) -> str:
    payload = json.dumps([asdict(row) for row in manifest], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_episode(row: EpisodeManifest, method: str) -> dict[str, Any]:
    if method not in {"no_adaptation", "oracle"}:
        raise ValueError(f"Unknown method: {method}")
    physics = Physics(row.mass, row.friction)
    env = make_env(row.task, physics=physics, seed=row.seed * 100_000 + row.index)
    observation = env.reset()
    policy = FrozenNominalPolicy(row.task, env.config)
    mapper = AdapterActionMapper(load_yaml(f"configs/adapter/{row.task}.yaml"))
    oracle = OracleContextAdapter(row.task, enabled=method == "oracle")
    forces: list[float] = []
    saturated: list[bool] = []
    command_trace: list[np.ndarray] = []
    info: dict[str, Any] = {}
    for step in range(env.horizon):
        nominal_action = apply_behavior_style(
            policy(observation), row.behavior_policy_id, step
        )
        correction = oracle.correction(physics) if method == "oracle" else None
        applied = mapper.apply(nominal_action, correction)
        observation, _, terminated, truncated, info = env.step(
            applied.action, applied.controller
        )
        forces.append(float(info["contact_force"]))
        saturated.append(applied.saturated)
        command_trace.append(applied.execution_command)
        if terminated or truncated:
            break
    result = {
        **asdict(row),
        "method": method,
        "action_trace_hash": execution_trace_hash(command_trace),
        **{key: value for key, value in info.items() if key != "contact_force"},
        **force_summary(forces, env.force_limit),
        "saturation_rate": float(np.mean(saturated)) if saturated else 0.0,
    }
    return result


def evaluate_manifest(manifest: list[EpisodeManifest], methods: Iterable[str]) -> dict[str, Any]:
    methods = list(methods)
    rows = [run_episode(item, method) for method in methods for item in manifest]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["method"]), str(row["split"]))].append(row)
    summary = {
        f"{task}/{method}/{split}": aggregate_episode_metrics(group)
        for (task, method, split), group in sorted(grouped.items())
    }
    return {"manifest_hash": manifest_hash(manifest), "summary": summary, "episodes": rows}


def gate_decision(result: dict[str, Any]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for task in sorted({row["task"] for row in result["episodes"]}):
        task_rows = [row for row in result["episodes"] if row["task"] == task]
        task_result = evaluate_rows(task_rows)
        nominal = task_result["no_adaptation/nominal"]["success_rate"]
        shifted = [
            task_result[f"no_adaptation/{split}"]["success_rate"]
            for split in ("ood_mass", "ood_friction", "ood_composition")
        ]
        oracle_shifted = [
            task_result[f"oracle/{split}"]["success_rate"]
            for split in ("ood_mass", "ood_friction", "ood_composition")
        ]
        informative = nominal - min(shifted) >= 0.20
        recovery = float(np.mean(oracle_shifted)) - float(np.mean(shifted)) >= 0.10
        safe = all(
            row["saturation_rate"] == 0.0 and row["force_violation_rate"] == 0.0
            for row in task_rows
            if row["method"] == "oracle"
        )
        decisions[task] = {
            "nominal_competence": nominal >= 0.8,
            "informative_shift": informative,
            "oracle_recovery": recovery,
            "oracle_bounds_and_force": safe,
        }
    passed = all(all(values.values()) for values in decisions.values())
    return {"passed": passed, "tasks": decisions}


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["split"])].append(row)
    return {
        f"{method}/{split}": aggregate_episode_metrics(group)
        for (method, split), group in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the oracle-first benchmark viability gate")
    parser.add_argument("--task", choices=["push", "lift", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_gate.json"))
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if the gate fails")
    args = parser.parse_args()
    tasks = ["push", "lift"] if args.task == "all" else [args.task]
    manifest = [row for task in tasks for row in make_manifest(task, args.seeds, args.episodes)]
    result = evaluate_manifest(manifest, ["no_adaptation", "oracle"])
    result["gate"] = gate_decision(result)
    result["backend_warning"] = (
        "Synthetic infrastructure result only; not evidence for a robotics or representation claim."
    )
    result["reproducibility"] = {
        "git_commit": git_commit(),
        "seeds": args.seeds,
        "physics_configuration": ["configs/physics/train.yaml", "configs/physics/ood.yaml"],
        "base_policy_checkpoint": "synthetic_visual_servo_v1",
        "encoder_checkpoint": None,
        "adapter_checkpoint": "engineering_oracle_v1",
        "normalization_statistics": "identity",
        "evaluation_splits": [item.value for item in PhysicsSplit] + ["nominal"],
        "versions": runtime_versions(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_hash": result["manifest_hash"], "gate": result["gate"]}, indent=2))
    if args.strict and not result["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
