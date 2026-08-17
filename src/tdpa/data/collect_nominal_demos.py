"""Collect privileged-expert labels over deployment-only observations.

This collector intentionally stores no object pose, contact state, physics value, or
task identifier in model arrays. Such values may appear only in archive-level audit
metadata and never leave :class:`NominalActionChunkDataset`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.data.nominal_demonstrations import (
    SCHEMA_VERSION,
    NominalDemonstrationArchive,
    file_sha256,
)
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.policies.frozen_nominal import assert_deployment_observation
from tdpa.policies.privileged_expert import PrivilegedScriptedExpert
from tdpa.utils.config import load_yaml


def collect_nominal_demonstrations(
    task: str,
    *,
    episodes: int,
    seed: int,
    output: Path,
    min_success_rate: float = 0.95,
    min_eligible_rate: float = 0.8,
    collection_split: str = "train",
    index_start: int = 0,
) -> dict[str, Any]:
    if task not in {"push", "lift"}:
        raise ValueError("task must be push or lift")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be in [0, 1]")
    if not 0.0 <= min_eligible_rate <= 1.0:
        raise ValueError("min_eligible_rate must be in [0, 1]")
    if collection_split not in {"train", "smoke"}:
        raise ValueError("collection_split must be train or smoke")
    if index_start < 0:
        raise ValueError("index_start must be non-negative")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")

    train_physics = load_yaml("configs/physics/train.yaml")
    nominal = train_physics["nominal"]
    physics = Physics(float(nominal["mass"]), float(nominal["friction"]))
    config = load_yaml(f"configs/env/{task}.yaml")
    horizon = int(config["episode_length"])
    size = int(config["robosuite"]["image_size"])
    rgb = np.zeros((episodes, horizon, 3, size, size), dtype=np.uint8)
    depth = np.zeros((episodes, horizon, 1, size, size), dtype=np.float16)
    proprio = np.zeros((episodes, horizon, 10), dtype=np.float32)
    actions = np.zeros((episodes, horizon, 4), dtype=np.float32)
    valid = np.zeros((episodes, horizon), dtype=np.bool_)
    terminals = np.zeros((episodes, horizon), dtype=np.bool_)
    success = np.zeros(episodes, dtype=np.bool_)
    eligible = np.zeros(episodes, dtype=np.bool_)
    episode_ids = index_start + np.arange(episodes, dtype=np.int64)
    summaries: list[dict[str, Any]] = []
    versions: dict[str, str] = {}

    for episode in range(episodes):
        env = make_env(
            task,
            physics=physics,
            seed=seed,
            episode_index=int(episode_ids[episode]),
            backend="robosuite",
            config=config,
        )
        expert = PrivilegedScriptedExpert(task)
        phase_counts: Counter[str] = Counter()
        maximum_force = 0.0
        saturation_steps = 0
        try:
            observation = env.reset()
            assert_deployment_observation(observation)
            reset_fingerprint = env.reset_fingerprint()
            info = env.metrics()
            for step in range(horizon):
                decision = expert.act(env, observation)
                image = np.asarray(observation["rgbd"], dtype=np.float32)
                state = np.asarray(observation["proprio"], dtype=np.float32)
                if image.shape != (4, size, size) or state.shape != (10,):
                    raise RuntimeError("robosuite observation does not match archive schema")
                rgb[episode, step] = np.rint(np.clip(image[:3], 0.0, 1.0) * 255.0).astype(np.uint8)
                depth[episode, step] = image[3:].astype(np.float16)
                proprio[episode, step] = state
                actions[episode, step] = decision.action
                valid[episode, step] = True
                phase_counts[decision.phase] += 1
                observation, _, terminated, truncated, info = env.step(decision.action)
                maximum_force = max(maximum_force, float(info["contact_force"]))
                saturation_steps += int(bool(info["controller_saturated"]))
                if terminated or truncated:
                    break
            success[episode] = bool(info["success"])
            terminals[episode, int(valid[episode].sum()) - 1] = True
            force_violation = bool(maximum_force > env.force_limit)
            # Push reports object-table force to audit the randomized friction interface;
            # it is not a robot-object safety metric and must not silently censor demos.
            force_eligible = task == "push" or not force_violation
            eligible[episode] = success[episode] and force_eligible and saturation_steps == 0
            versions = env.versions()
            summaries.append(
                {
                    "episode_id": int(episode_ids[episode]),
                    "length": int(valid[episode].sum()),
                    "success": bool(success[episode]),
                    "reset_fingerprint": reset_fingerprint,
                    "maximum_contact_force": maximum_force,
                    "force_violation": force_violation,
                    "controller_saturation_steps": saturation_steps,
                    "eligible_for_training": bool(eligible[episode]),
                    "phase_counts": dict(sorted(phase_counts.items())),
                    "expert_grasp_losses": expert.grasp_losses,
                    "failure_reason": None if success[episode] else "timeout_or_task_failure",
                }
            )
        finally:
            env.close()

    success_rate = float(success.mean())
    eligible_rate = float(eligible.mean())
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "backend": "robosuite",
        "expert_version": PrivilegedScriptedExpert.version,
        "observation_keys": ["rgbd", "proprio"],
        "action_schema": ["delta_x", "delta_y", "delta_z", "gripper"],
        "controller_mode": "fixed_nominal",
        "physics": {"mass": physics.mass, "friction": physics.friction},
        "environment_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "seed": seed,
        "collection_split": collection_split,
        "reset_index_start": index_start,
        "reset_index_end_exclusive": index_start + episodes,
        "episodes": episodes,
        "horizon": horizon,
        "success_rate": success_rate,
        "python": platform.python_version(),
        "versions": versions,
        "episode_summaries": summaries,
        "leakage_boundary": (
            "privileged state labels expert actions only; model arrays contain RGB-D, "
            "proprioception, actions, and validity"
        ),
        "training_eligibility_rule": (
            "success and no controller saturation; Lift additionally requires no fingerpad "
            "force-limit violation. Push object-table force is diagnostic only."
        ),
    }
    archive = NominalDemonstrationArchive(
        rgb=rgb,
        depth=depth,
        proprio=proprio,
        actions=actions,
        valid=valid,
        terminals=terminals,
        success=success,
        eligible=eligible,
        episode_ids=episode_ids,
        metadata=metadata,
    )
    archive.save(output)
    result = {
        "status": (
            "PASS"
            if success_rate >= min_success_rate and eligible_rate >= min_eligible_rate
            else "FAIL"
        ),
        "task": task,
        "output": str(output),
        "sha256": file_sha256(output),
        "episodes": episodes,
        "successful_episodes": int(success.sum()),
        "success_rate": success_rate,
        "eligible_episodes": int(eligible.sum()),
        "eligible_rate": eligible_rate,
        "minimum_success_rate": min_success_rate,
        "minimum_eligible_rate": min_eligible_rate,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise RuntimeError(
            "Expert collection quality is below threshold: "
            f"success={success_rate:.1%}/{min_success_rate:.1%}, "
            f"eligible={eligible_rate:.1%}/{min_eligible_rate:.1%}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("push", "lift"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-eligible-rate", type=float, default=0.8)
    parser.add_argument("--collection-split", choices=("train", "smoke"), default="train")
    parser.add_argument("--index-start", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect_nominal_demonstrations(
        args.task,
        episodes=args.episodes,
        seed=args.seed,
        output=args.output,
        min_success_rate=args.min_success_rate,
        min_eligible_rate=args.min_eligible_rate,
        collection_split=args.collection_split,
        index_start=args.index_start,
    )


if __name__ == "__main__":
    main()
