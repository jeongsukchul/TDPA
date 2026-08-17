"""No-training smoke for expert data, BC tensors, strict loading, and live execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tdpa.data.collect_nominal_demos import collect_nominal_demonstrations
from tdpa.data.nominal_demonstrations import (
    NominalActionChunkDataset,
    NominalDemonstrationArchive,
    episode_split,
    fit_proprio_normalization,
)
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.evaluation.evaluate_nominal_policy import evaluate_nominal_policy
from tdpa.models.nominal_bc import VisualActionChunkBC
from tdpa.policies.learned_nominal import (
    FrozenLearnedNominalPolicy,
    checkpoint_sha256,
    save_nominal_checkpoint,
)
from tdpa.utils.config import load_yaml


def smoke_task(task: str, *, seed: int, directory: Path) -> dict[str, object]:
    dataset_path = directory / f"{task}_untrained_smoke_demos.hdf5"
    checkpoint_path = directory / f"{task}_untrained_smoke.pt"
    collect_nominal_demonstrations(
        task,
        episodes=6,
        seed=seed,
        output=dataset_path,
        min_success_rate=0.95,
        min_eligible_rate=0.3,
        collection_split="smoke",
        index_start=30_000,
    )
    archive = NominalDemonstrationArchive.load(dataset_path)
    split = episode_split(archive, validation_fraction=0.5, seed=seed)
    normalization = fit_proprio_normalization(archive, split["train"])
    dataset = NominalActionChunkDataset(
        archive,
        split["train"],
        history_length=2,
        action_horizon=8,
        normalization=normalization,
    )
    sample = dataset[0]
    model = VisualActionChunkBC(history_length=2, action_horizon=8)
    with torch.inference_mode():
        prediction = model(
            sample["rgbd_history"].unsqueeze(0),
            sample["proprio_history"].unsqueeze(0),
            sample["observation_mask"].unsqueeze(0),
        )
    if prediction.shape != (1, 8, 4) or not torch.isfinite(prediction).all():
        raise RuntimeError("BC tensor smoke failed")
    save_nominal_checkpoint(
        checkpoint_path,
        model=model,
        task=task,
        normalization=normalization,
        status="untrained_smoke",
        provenance={
            "training_steps": 0,
            "purpose": "interface_smoke_only",
            "eligible_for_results": False,
        },
    )
    strict_rejection = False
    try:
        FrozenLearnedNominalPolicy(checkpoint_path, task=task)
    except RuntimeError:
        strict_rejection = True
    if not strict_rejection:
        raise RuntimeError("Production loader accepted an untrained checkpoint")
    policy = FrozenLearnedNominalPolicy(
        checkpoint_path, task=task, allow_untrained=True, device="cpu"
    )
    if not policy.frozen:
        raise RuntimeError("Loaded nominal model is not frozen")

    nominal = load_yaml("configs/physics/train.yaml")["nominal"]
    env = make_env(
        task,
        physics=Physics(float(nominal["mass"]), float(nominal["friction"])),
        seed=seed + 1,
        episode_index=0,
        backend="robosuite",
    )
    try:
        observation = env.reset()
        policy.reset()
        action_norms: list[float] = []
        for _ in range(2):
            action = policy(observation)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise RuntimeError("Frozen policy emitted an invalid live action")
            action_norms.append(float(np.linalg.norm(action)))
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        bounds = env.robosuite_config["execution"]["bounds"]
        low_controller = {key: float(value[0]) for key, value in bounds.items()}
        env.step(np.zeros(4, dtype=np.float32), low_controller)
        low_readback = env.controller_readback()
        high_controller = {key: float(value[1]) for key, value in bounds.items()}
        env.step(np.zeros(4, dtype=np.float32), high_controller)
        high_readback = env.controller_readback()
        env.reset()
        reset_readback = env.controller_readback()
    finally:
        env.close()
    evaluation_path = directory / f"{task}_untrained_evaluator_smoke.json"
    evaluation = evaluate_nominal_policy(
        argparse.Namespace(
            mode="smoke",
            task=task,
            checkpoint=checkpoint_path,
            competence_artifact=None,
            output=evaluation_path,
            seeds=[seed + 2],
            episodes=1,
            device="cpu",
        )
    )
    return {
        "task": task,
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_status": "untrained_smoke",
        "training_steps": 0,
        "eligible_for_results": False,
        "strict_loader_rejected_untrained": strict_rejection,
        "frozen_loader": policy.frozen,
        "model_output_shape": list(prediction.shape),
        "dataset_item_keys": sorted(sample),
        "live_action_norms": action_norms,
        "low_controller": low_readback,
        "high_controller": high_readback,
        "reset_controller": reset_readback,
        "evaluator_smoke": {
            "status": evaluation["status"],
            "cells": evaluation["cells"],
            "manifest_sha256": evaluation["manifest_sha256"],
            "output": str(evaluation_path),
        },
    }


def run_smoke(*, tasks: list[str], seed: int, output: Path) -> dict[str, object]:
    directory = output.parent / "nominal_policy_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [smoke_task(task, seed=seed, directory=directory) for task in tasks]
    report: dict[str, object] = {
        "status": "PASS",
        "scope": "no-training nominal-policy plumbing smoke",
        "warning": "Untrained checkpoints are rejected by production loaders and are not results.",
        "tasks": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("push", "lift", "all"), default="all")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, default=Path("artifacts/nominal_policy_smoke.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = ["push", "lift"] if args.task == "all" else [args.task]
    run_smoke(tasks=tasks, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
