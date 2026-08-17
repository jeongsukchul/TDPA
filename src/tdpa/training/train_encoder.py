from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from tdpa.data.interaction_collector import load_episode_archive
from tdpa.data.sequence_dataset import SequenceDataset
from tdpa.models.bundle import EncoderBundle
from tdpa.models.response_predictor import normalized_mse
from tdpa.training.datasets import PrivilegedTeacherDataset
from tdpa.utils.checkpoints import file_sha256, save_checkpoint
from tdpa.utils.config import load_yaml
from tdpa.utils.seed import seed_everything


def _make_dataset(
    paths: list[Path], config: dict[str, Any]
) -> tuple[ConcatDataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    datasets = []
    window_probes: list[str] = []
    response_values: list[torch.Tensor] = []
    for path in paths:
        episodes = load_episode_archive(path)
        if any(episode.metadata.get("split") != "train" for episode in episodes):
            raise RuntimeError("Encoder pretraining accepts only interaction split='train'")
        response_values.extend(episode.responses[1:].float() for episode in episodes)
        # Balanced collection is asserted before batching, keeping probe/action
        # frequency independent of physics sample count.
        probe_counts: dict[str, int] = {}
        for episode in episodes:
            probe = str(episode.metadata.get("probe_id"))
            probe_counts[probe] = probe_counts.get(probe, 0) + 1
        if probe_counts and max(probe_counts.values()) - min(probe_counts.values()) > 1:
            raise RuntimeError(f"Probe primitives are not balanced in {path}: {probe_counts}")
        safe = SequenceDataset(
            episodes,
            history_length=int(config["history_length"]),
            future_horizon=int(config["future_length"]),
        )
        for item in range(len(safe)):
            episode_index = safe.window_alignment(item).episode_index
            window_probes.append(str(episodes[episode_index].metadata.get("probe_id")))
        datasets.append(PrivilegedTeacherDataset(safe))
    counts = {probe: window_probes.count(probe) for probe in set(window_probes)}
    weights = torch.tensor([1.0 / counts[probe] for probe in window_probes], dtype=torch.double)
    stacked_response = torch.cat(response_values)
    response_mean = stacked_response.mean(dim=0)
    response_std = stacked_response.std(dim=0, unbiased=False).clamp_min(1e-4)
    return ConcatDataset(datasets), weights, response_mean, response_std


def compute_loss(
    bundle: EncoderBundle,
    batch: dict[str, torch.Tensor],
    variant: str,
    weight: float,
    response_mean: torch.Tensor,
    response_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgbd = batch["rgbd_history"] if bundle.config.get("use_rgbd", True) else None
    mask = batch["history_mask"]
    future_mask = batch["future_mask"]
    student = bundle.student(rgbd, batch["proprio_history"], batch["action_history"], mask)
    teacher = bundle.teacher(batch["privileged_history"], mask)
    # E_R is a fixed non-collapsed target transform in this MVP. It is never
    # updated through the predictor objective.
    with torch.no_grad():
        normalized_response = (
            batch["future_response_sequence"] - response_mean.view(1, 1, -1)
        ) / response_std.view(1, 1, -1)
        target = bundle.response_encoder(normalized_response, future_mask)
    student_prediction = bundle.student_predictor(
        student, batch["future_action_sequence"], future_mask
    )
    teacher_prediction = bundle.teacher_predictor(
        teacher, batch["future_action_sequence"], future_mask
    )
    student_response = normalized_mse(student_prediction, target)
    teacher_response = normalized_mse(teacher_prediction, target)
    distill = F.mse_loss(F.normalize(student, dim=-1), F.normalize(teacher.detach(), dim=-1))
    if variant == "response":
        loss = student_response
    elif variant == "distill":
        loss = teacher_response + weight * distill
    elif variant == "hybrid":
        loss = 0.5 * (student_response + teacher_response) + weight * distill
    else:
        raise ValueError(f"Unknown representation variant: {variant}")
    metrics = {
        "loss": float(loss.detach()),
        "student_response": float(student_response.detach()),
        "teacher_response": float(teacher_response.detach()),
        "distill": float(distill.detach()),
    }
    return loss, metrics


def train(
    *,
    variant: str,
    datasets: list[Path],
    output: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    balanced: bool = True,
) -> dict[str, float]:
    seed_everything(seed)
    config = load_yaml(f"configs/encoder/{variant}.yaml")
    data, weights, response_mean, response_std = _make_dataset(datasets, config)
    config["normalization"] = {
        "response_mean": response_mean.tolist(),
        "response_std": response_std.tolist(),
        "fit_split": "train",
    }
    sampler = None
    if balanced:
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    loader = DataLoader(data, batch_size=batch_size, sampler=sampler, shuffle=not balanced)
    bundle = EncoderBundle(config).to(device)
    response_mean = response_mean.to(device)
    response_std = response_std.to(device)
    for parameter in bundle.response_encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in bundle.parameters() if parameter.requires_grad], lr=3e-4
    )
    final: dict[str, float] = {}
    bundle.train()
    bundle.response_encoder.eval()
    for _ in range(epochs):
        totals: dict[str, float] = {}
        steps = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, metrics = compute_loss(
                bundle,
                batch,
                variant,
                float(config.get("lambda_distill", 1.0)),
                response_mean,
                response_std,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), 5.0)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        final = {key: value / max(steps, 1) for key, value in totals.items()}
    save_checkpoint(
        output,
        model=bundle,
        config=config,
        optimizer=optimizer,
        metadata={
            "variant": variant,
            "seed": seed,
            "epochs": epochs,
            "dataset_hashes": {str(path): file_sha256(path) for path in datasets},
            "normalization_statistics": config["normalization"],
            "deployment_inputs": ["rgbd", "proprio", "action"],
            "privileged_inputs": ["teacher_history"],
            "final_metrics": final,
            "probe_balanced_sampling": balanced,
        },
    )
    student_output = output.with_name(f"{output.stem}_student{output.suffix}")
    save_checkpoint(
        student_output,
        model=bundle.student,
        config=config,
        metadata={
            "artifact_type": "deployment_student",
            "variant": variant,
            "seed": seed,
            "source_training_checkpoint_hash": file_sha256(output),
            "deployment_inputs": ["rgbd_without_goal_channel", "proprio", "execution_command"],
            "privileged_inputs": [],
        },
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Train response, distillation, or hybrid encoder")
    parser.add_argument("--variant", choices=["response", "distill", "hybrid"], required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--unbalanced",
        dest="balanced",
        action="store_false",
        help="Anti-shortcut ablation: disable probe-balanced sampling",
    )
    parser.set_defaults(balanced=True)
    args = parser.parse_args()
    metrics = train(**vars(args))
    print({key: round(value, 6) for key, value in metrics.items()})


if __name__ == "__main__":
    main()
