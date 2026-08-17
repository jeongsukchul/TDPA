"""Train a task-specific visual action-chunk behavior-cloning policy."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from tdpa.data.nominal_demonstrations import (
    NominalActionChunkDataset,
    NominalDemonstrationArchive,
    episode_split,
    file_sha256,
    fit_proprio_normalization,
    split_manifest_sha256,
)
from tdpa.models.nominal_bc import VisualActionChunkBC
from tdpa.policies.learned_nominal import (
    checkpoint_sha256,
    current_environment_hash,
    save_nominal_checkpoint,
)
from tdpa.utils.config import load_yaml


def _masked_action_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gripper_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("Action prediction, target, and mask shapes do not align")
    weights = mask.to(dtype=prediction.dtype)
    denominator = weights.sum().clamp_min(1.0)
    motion_per_step = functional.mse_loss(
        prediction[..., :3], target[..., :3], reduction="none"
    ).mean(dim=-1)
    gripper_per_step = functional.mse_loss(prediction[..., 3], target[..., 3], reduction="none")
    motion = (motion_per_step * weights).sum() / denominator
    gripper = (gripper_per_step * weights).sum() / denominator
    return motion + gripper_weight * gripper, motion, gripper


def _masked_action_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return _masked_action_losses(prediction, target, mask, gripper_weight=1.0)[0]


def _epoch(
    model: VisualActionChunkBC,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gripper_weight: float,
) -> tuple[float, float, float]:
    model.train(optimizer is not None)
    total = 0.0
    total_motion = 0.0
    total_gripper = 0.0
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            rgbd = batch["rgbd_history"].to(device)
            proprio = batch["proprio_history"].to(device)
            observation_mask = batch["observation_mask"].to(device)
            target = batch["action_chunk"].to(device)
            action_mask = batch["action_mask"].to(device)
            prediction = model(rgbd, proprio, observation_mask)
            loss, motion_loss, gripper_loss = _masked_action_losses(
                prediction,
                target,
                action_mask,
                gripper_weight=gripper_weight,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = int(rgbd.shape[0])
            total += float(loss.detach()) * batch_size
            total_motion += float(motion_loss.detach()) * batch_size
            total_gripper += float(gripper_loss.detach()) * batch_size
            count += batch_size
    if count == 0:
        raise RuntimeError("Data loader produced no batches")
    return total / count, total_motion / count, total_gripper / count


def train_nominal_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.gripper_weight <= 0:
        raise ValueError("gripper_weight must be positive")
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    device = torch.device(device_name)

    archive = NominalDemonstrationArchive.load(args.dataset)
    if archive.metadata.get("task") != args.task:
        raise ValueError("Dataset task does not match requested checkpoint task")
    if archive.metadata.get("backend") != "robosuite":
        raise ValueError("Nominal policy training accepts only robosuite demonstrations")
    if archive.metadata.get("collection_split") != "train":
        raise ValueError("Nominal policy training accepts only declared train archives")
    nominal = load_yaml("configs/physics/train.yaml")["nominal"]
    archive_physics = archive.metadata.get("physics")
    expected_physics = {"mass": float(nominal["mass"]), "friction": float(nominal["friction"])}
    if archive_physics != expected_physics:
        raise ValueError("Demonstration physics does not match the locked nominal configuration")
    if archive.metadata.get("environment_hash") != current_environment_hash(args.task):
        raise ValueError("Demonstration environment hash does not match current task config")
    if np.any(archive.episode_ids < 0) or np.any(archive.episode_ids >= 20_000):
        raise ValueError("Training reset indexes must stay in the reserved [0, 20000) namespace")
    split = episode_split(
        archive,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        successful_only=True,
    )
    normalization = fit_proprio_normalization(archive, split["train"])
    train_dataset = NominalActionChunkDataset(
        archive,
        split["train"],
        history_length=args.history_length,
        action_horizon=args.action_horizon,
        normalization=normalization,
    )
    validation_dataset = NominalActionChunkDataset(
        archive,
        split["validation"],
        history_length=args.history_length,
        action_horizon=args.action_horizon,
        normalization=normalization,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = VisualActionChunkBC(
        history_length=args.history_length,
        action_horizon=args.action_horizon,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_motion, train_gripper = _epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            gripper_weight=args.gripper_weight,
        )
        validation_loss, validation_motion, validation_gripper = _epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            gripper_weight=args.gripper_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_motion_mse": train_motion,
                "train_gripper_mse": train_gripper,
                "validation_loss": validation_loss,
                "validation_motion_mse": validation_motion,
                "validation_gripper_mse": validation_gripper,
            }
        )
        print(json.dumps(history[-1], sort_keys=True))
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    provenance = {
        "training_steps": len(train_loader) * args.epochs,
        "epochs": args.epochs,
        "best_validation_loss": best_validation,
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "split_sha256": split_manifest_sha256(archive, split),
        "train_episode_ids": archive.episode_ids[split["train"]].astype(int).tolist(),
        "validation_episode_ids": archive.episode_ids[split["validation"]].astype(int).tolist(),
        "seed": args.seed,
        "collection_seed": int(archive.metadata["seed"]),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gripper_weight": args.gripper_weight,
        "wall_time_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "attempted_episodes": archive.episode_count,
        "successful_episodes": int(archive.success.sum()),
        "competence_gate": "not_run",
        "training_history": history,
    }
    save_nominal_checkpoint(
        args.output,
        model=model,
        task=args.task,
        normalization=normalization,
        status="trained",
        provenance=provenance,
    )
    result = {
        "status": "TRAINED_NOT_EVALUATED",
        "task": args.task,
        "checkpoint": str(args.output),
        "checkpoint_sha256": checkpoint_sha256(args.output),
        "best_validation_loss": best_validation,
        "warning": "Loss is not task competence; run the locked closed-loop nominal gate.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("push", "lift"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    train_nominal_policy(build_parser().parse_args())


if __name__ == "__main__":
    main()
