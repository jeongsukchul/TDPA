from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tdpa.data.interaction_collector import load_episode_archive
from tdpa.data.sequence_dataset import SequenceDataset
from tdpa.models.bundle import load_encoder_bundle
from tdpa.models.response_predictor import normalized_mse
from tdpa.utils.checkpoints import file_sha256


@torch.no_grad()
def encode_dataset(
    bundle_path: Path, dataset_path: Path, device: str = "cpu"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bundle = load_encoder_bundle(str(bundle_path), device)
    encoder = bundle.freeze_deployment_encoder()
    episodes = load_episode_archive(dataset_path)
    dataset = SequenceDataset(
        episodes,
        history_length=int(bundle.config["history_length"]),
        future_horizon=int(bundle.config["future_length"]),
        allow_partial_windows=False,
    )
    latents = []
    physics = []
    probe_ids = []
    policy_ids = []
    task_ids = []
    physics_keys = []
    for item_index, batch in enumerate(DataLoader(dataset, batch_size=128, shuffle=False)):
        batch = {key: value.to(device) for key, value in batch.items()}
        rgbd = batch["rgbd_history"] if bundle.config.get("use_rgbd", True) else None
        latents.append(
            encoder(rgbd, batch["proprio_history"], batch["action_history"], batch["history_mask"]).cpu()
        )
        start = item_index * 128
        for index in range(start, min(start + len(next(iter(batch.values()))), len(dataset))):
            alignment = dataset.window_alignment(index)
            metadata = episodes[alignment.episode_index].metadata
            physics.append([metadata["physics"]["mass"], metadata["physics"]["friction"]])
            probe_ids.append(metadata["probe_id"])
            policy_ids.append(metadata["policy_id"])
            task_ids.append(metadata["task"])
            physics_keys.append(
                f"{metadata['task']}|{metadata['physics']['mass']:.9f}:"
                f"{metadata['physics']['friction']:.9f}"
            )
    return torch.cat(latents).numpy(), {
        "physics": np.asarray(physics, dtype=np.float32),
        "probe_id": np.asarray(probe_ids),
        "policy_id": np.asarray(policy_ids),
        "task_id": np.asarray(task_ids),
        "physics_key": np.asarray(physics_keys),
    }


def linear_regression_score(features: np.ndarray, targets: np.ndarray) -> list[float | None]:
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    weights = np.linalg.lstsq(design, targets, rcond=None)[0]
    predictions = design @ weights
    residual = np.square(targets - predictions).sum(axis=0)
    total = np.square(targets - targets.mean(axis=0)).sum(axis=0)
    return [
        None
        if float(np.ptp(targets[:, axis])) < 1e-6
        else float(1.0 - residual[axis] / total[axis])
        for axis in range(targets.shape[1])
    ]


def nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float | None:
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    centroids = np.stack([features[labels == label].mean(axis=0) for label in classes])
    predictions = classes[np.square(features[:, None] - centroids[None]).sum(axis=-1).argmin(axis=1)]
    return float(np.mean(predictions == labels))


def heldout_physics_centroid_accuracy(
    features: np.ndarray, labels: np.ndarray, physics_keys: np.ndarray
) -> float | None:
    physics_values = np.unique(physics_keys)
    if len(physics_values) < 2 or len(np.unique(labels)) < 2:
        return None
    train_physics = set(physics_values[::2])
    train_mask = np.asarray([value in train_physics for value in physics_keys])
    test_mask = ~train_mask
    classes = np.unique(labels)
    if not test_mask.any() or any(not np.any(train_mask & (labels == label)) for label in classes):
        return None
    centroids = np.stack([features[train_mask & (labels == label)].mean(axis=0) for label in classes])
    predictions = classes[
        np.square(features[test_mask, None] - centroids[None]).sum(axis=-1).argmin(axis=1)
    ]
    return float(np.mean(predictions == labels[test_mask]))


def cross_policy_retrieval(
    features: np.ndarray, physics_keys: np.ndarray, policy_ids: np.ndarray
) -> dict[str, float | None]:
    means: dict[tuple[str, str], np.ndarray] = {}
    for physics in np.unique(physics_keys):
        for policy in np.unique(policy_ids[physics_keys == physics]):
            mask = (physics_keys == physics) & (policy_ids == policy)
            means[(str(physics), str(policy))] = features[mask].mean(axis=0)
    same = []
    for physics in np.unique(physics_keys):
        policies = sorted({policy for key, policy in means if key == physics})
        if len(policies) >= 2:
            same.append(float(np.linalg.norm(means[(physics, policies[0])] - means[(physics, policies[1])])))
    different = []
    keys = sorted(means)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            same_task = left[0].split("|", 1)[0] == right[0].split("|", 1)[0]
            if same_task and left[0] != right[0]:
                different.append(float(np.linalg.norm(means[left] - means[right])))
    same_mean = float(np.mean(same)) if same else None
    different_mean = float(np.mean(different)) if different else None
    ratio = (
        same_mean / different_mean
        if same_mean is not None and different_mean not in (None, 0.0)
        else None
    )
    return {
        "same_physics_cross_policy_distance": same_mean,
        "different_physics_distance": different_mean,
        "cross_policy_distance_ratio": ratio,
    }


@torch.no_grad()
def response_diagnostics(
    bundle_path: Path, dataset_paths: list[Path], device: str
) -> dict[str, float]:
    bundle = load_encoder_bundle(str(bundle_path), device).eval()
    normalization = bundle.config.get("normalization", {})
    response_mean = torch.tensor(
        normalization.get("response_mean", [0.0] * int(bundle.config["response_dim"])),
        device=device,
    ).view(1, 1, -1)
    response_std = torch.tensor(
        normalization.get("response_std", [1.0] * int(bundle.config["response_dim"])),
        device=device,
    ).view(1, 1, -1)
    losses = []
    shuffled_losses = []
    zero_latent_losses = []
    action_effects = []
    normalized_targets = []
    for dataset_path in dataset_paths:
        episodes = load_episode_archive(dataset_path)
        dataset = SequenceDataset(
            episodes,
            history_length=int(bundle.config["history_length"]),
            future_horizon=int(bundle.config["future_length"]),
            allow_partial_windows=False,
        )
        for batch in DataLoader(dataset, batch_size=128, shuffle=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            rgbd = batch["rgbd_history"] if bundle.config.get("use_rgbd", True) else None
            latent = bundle.student(
                rgbd, batch["proprio_history"], batch["action_history"], batch["history_mask"]
            )
            target = bundle.response_encoder(
                (batch["future_response_sequence"] - response_mean) / response_std,
                batch["future_mask"],
            )
            prediction = bundle.student_predictor(
                latent, batch["future_action_sequence"], batch["future_mask"]
            )
            shuffled = bundle.student_predictor(
                latent.flip(0), batch["future_action_sequence"], batch["future_mask"]
            )
            zero_latent = bundle.student_predictor(
                torch.zeros_like(latent),
                batch["future_action_sequence"],
                batch["future_mask"],
            )
            counterfactual = bundle.student_predictor(
                latent, torch.zeros_like(batch["future_action_sequence"]), batch["future_mask"]
            )
            losses.append(float(normalized_mse(prediction, target)))
            shuffled_losses.append(float(normalized_mse(shuffled, target)))
            zero_latent_losses.append(float(normalized_mse(zero_latent, target)))
            normalized_targets.append(torch.nn.functional.normalize(target, dim=-1).cpu())
            action_effects.append(float(torch.mean(torch.linalg.vector_norm(prediction - counterfactual, dim=-1))))
    all_targets = torch.cat(normalized_targets)
    constant = torch.nn.functional.normalize(all_targets.mean(dim=0), dim=0)
    constant_mse = float(torch.mean(torch.square(all_targets - constant)))
    return {
        "response_normalized_mse_on_supplied_data": float(np.mean(losses)),
        "shuffled_latent_normalized_mse": float(np.mean(shuffled_losses)),
        "zero_latent_normalized_mse": float(np.mean(zero_latent_losses)),
        "constant_target_normalized_mse": constant_mse,
        "counterfactual_action_embedding_change": float(np.mean(action_effects)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen representation diagnostics")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-heldout", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/representation_probe.json"))
    args = parser.parse_args()
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    training_hashes = set(checkpoint.get("metadata", {}).get("dataset_hashes", {}).values())
    supplied_hashes = {file_sha256(path) for path in args.dataset}
    overlap = bool(training_hashes & supplied_hashes)
    if args.require_heldout and overlap:
        raise RuntimeError("Representation diagnostics must not reuse encoder-training archives")
    encoded = [encode_dataset(args.encoder, path, args.device) for path in args.dataset]
    features = np.concatenate([item[0] for item in encoded])
    labels = {
        key: np.concatenate([item[1][key] for item in encoded]) for key in encoded[0][1]
    }
    physics_r2 = linear_regression_score(features, labels["physics"])
    result = {
        "mass_linear_probe_r2": physics_r2[0],
        "friction_linear_probe_r2": physics_r2[1],
        "probe_id_nearest_centroid_accuracy": nearest_centroid_accuracy(features, labels["probe_id"]),
        "policy_id_nearest_centroid_accuracy": nearest_centroid_accuracy(features, labels["policy_id"]),
        "task_id_nearest_centroid_accuracy": nearest_centroid_accuracy(features, labels["task_id"]),
        "probe_id_heldout_physics_accuracy": heldout_physics_centroid_accuracy(
            features, labels["probe_id"], labels["physics_key"]
        ),
        "policy_id_heldout_physics_accuracy": heldout_physics_centroid_accuracy(
            features, labels["policy_id"], labels["physics_key"]
        ),
        "task_id_heldout_physics_accuracy": heldout_physics_centroid_accuracy(
            features, labels["task_id"], labels["physics_key"]
        ),
        **cross_policy_retrieval(features, labels["physics_key"], labels["policy_id"]),
        **response_diagnostics(args.encoder, args.dataset, args.device),
        "probe_updates_encoder": False,
        "training_archive_overlap": overlap,
        "warning": (
            "Encoder archive overlap is checked, but centroid classifiers are resubstitution "
            "diagnostics; use episode/physics-disjoint classifier splits for reporting."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
