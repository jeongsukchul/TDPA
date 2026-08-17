from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SCHEMA_VERSION = "tdpa-nominal-demos-v1"


@dataclass(frozen=True)
class NominalDemonstrationArchive:
    rgb: np.ndarray
    depth: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    valid: np.ndarray
    terminals: np.ndarray
    success: np.ndarray
    eligible: np.ndarray
    episode_ids: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.valid.ndim != 2:
            raise ValueError("valid must have shape [episode, time]")
        if self.rgb.ndim != 5:
            raise ValueError("rgb must have shape [episode, time, channel, height, width]")
        episodes, horizon = self.valid.shape
        expected = {
            "rgb": (episodes, horizon, 3, self.rgb.shape[-2], self.rgb.shape[-1]),
            "depth": (episodes, horizon, 1, self.rgb.shape[-2], self.rgb.shape[-1]),
            "proprio": (episodes, horizon, 10),
            "actions": (episodes, horizon, 4),
            "terminals": (episodes, horizon),
            "success": (episodes,),
            "eligible": (episodes,),
            "episode_ids": (episodes,),
        }
        arrays = {
            "rgb": self.rgb,
            "depth": self.depth,
            "proprio": self.proprio,
            "actions": self.actions,
            "terminals": self.terminals,
            "success": self.success,
            "eligible": self.eligible,
            "episode_ids": self.episode_ids,
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} has shape {arrays[name].shape}; expected {shape}")
        if self.rgb.dtype != np.uint8:
            raise TypeError("rgb must be uint8")
        if self.depth.dtype != np.float16:
            raise TypeError("depth must be float16")
        if self.proprio.dtype != np.float32 or self.actions.dtype != np.float32:
            raise TypeError("proprio and actions must be float32")
        if not np.isfinite(self.depth).all() or not np.isfinite(self.proprio).all():
            raise ValueError("Depth and proprio arrays must be finite")
        if not np.isfinite(self.actions[self.valid]).all():
            raise ValueError("Valid actions must be finite")
        if np.any(self.depth < 0.0) or np.any(self.depth > 1.0):
            raise ValueError("Normalized depth must remain in [0, 1]")
        if np.any(self.actions[self.valid] < -1.0) or np.any(self.actions[self.valid] > 1.0):
            raise ValueError("Valid normalized actions must remain in [-1, 1]")
        if (
            self.valid.dtype != np.bool_
            or self.terminals.dtype != np.bool_
            or self.success.dtype != np.bool_
            or self.eligible.dtype != np.bool_
        ):
            raise TypeError("valid, terminals, success, and eligible must be boolean")
        if np.any(self.eligible & ~self.success):
            raise ValueError("Training-eligible episodes must be successful")
        if self.episode_ids.dtype != np.int64:
            raise TypeError("episode_ids must be int64")
        if np.any(self.valid[:, 1:] & ~self.valid[:, :-1]):
            raise ValueError("Valid timesteps must form a contiguous prefix per episode")
        if np.any(self.terminals & ~self.valid):
            raise ValueError("Terminal flags may only appear on valid timesteps")
        for episode in range(episodes):
            terminal_steps = np.flatnonzero(self.terminals[episode])
            if len(terminal_steps) != 1 or terminal_steps[0] != int(self.valid[episode].sum()) - 1:
                raise ValueError("Each episode must terminate exactly at its last valid timestep")
        if len(np.unique(self.episode_ids)) != episodes:
            raise ValueError("episode_ids must be unique")
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"metadata must declare {SCHEMA_VERSION}")

    @property
    def episode_count(self) -> int:
        return int(self.valid.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.valid.shape[1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "w") as handle:
                handle.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
                arrays = {
                    "rgb": self.rgb,
                    "depth": self.depth,
                    "proprio": self.proprio,
                    "actions": self.actions,
                    "valid": self.valid,
                    "terminals": self.terminals,
                    "success": self.success,
                    "eligible": self.eligible,
                    "episode_ids": self.episode_ids,
                }
                for name, array in arrays.items():
                    chunks = (1, *array.shape[1:]) if array.ndim > 1 else True
                    handle.create_dataset(
                        name,
                        data=array,
                        chunks=chunks,
                        compression="gzip",
                        shuffle=True,
                    )
            return
        np.savez_compressed(
            path,
            rgb=self.rgb,
            depth=self.depth,
            proprio=self.proprio,
            actions=self.actions,
            valid=self.valid,
            terminals=self.terminals,
            success=self.success,
            eligible=self.eligible,
            episode_ids=self.episode_ids,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> NominalDemonstrationArchive:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "r") as handle:
                metadata = json.loads(str(handle.attrs["metadata_json"]))
                return cls(
                    rgb=handle["rgb"][:],
                    depth=handle["depth"][:],
                    proprio=handle["proprio"][:],
                    actions=handle["actions"][:],
                    valid=handle["valid"][:],
                    terminals=handle["terminals"][:],
                    success=handle["success"][:],
                    eligible=handle["eligible"][:],
                    episode_ids=handle["episode_ids"][:],
                    metadata=metadata,
                )
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            return cls(
                rgb=data["rgb"].copy(),
                depth=data["depth"].copy(),
                proprio=data["proprio"].copy(),
                actions=data["actions"].copy(),
                valid=data["valid"].copy(),
                terminals=data["terminals"].copy(),
                success=data["success"].copy(),
                eligible=data["eligible"].copy(),
                episode_ids=data["episode_ids"].copy(),
                metadata=metadata,
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_split(
    archive: NominalDemonstrationArchive,
    *,
    validation_fraction: float = 0.15,
    seed: int = 0,
    successful_only: bool = True,
) -> dict[str, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    eligible = (
        np.flatnonzero(archive.eligible) if successful_only else np.arange(archive.episode_count)
    )
    if len(eligible) < 2:
        raise ValueError("At least two eligible episodes are required for a disjoint split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = np.sort(shuffled[:validation_count])
    train = np.sort(shuffled[validation_count:])
    if set(train) & set(validation):
        raise RuntimeError("Episode-level train/validation split overlap")
    return {"train": train, "validation": validation}


def split_manifest_sha256(
    archive: NominalDemonstrationArchive, split: dict[str, np.ndarray]
) -> str:
    payload = {
        name: archive.episode_ids[indexes].astype(int).tolist()
        for name, indexes in sorted(split.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_proprio_normalization(
    archive: NominalDemonstrationArchive, episode_indexes: np.ndarray
) -> dict[str, list[float]]:
    selected = archive.proprio[episode_indexes]
    mask = archive.valid[episode_indexes]
    values = selected[mask]
    if values.size == 0:
        raise ValueError("Cannot fit normalization on an empty split")
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return {"mean": mean.tolist(), "std": std.tolist()}


class NominalActionChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal observation histories paired with within-episode future actions."""

    def __init__(
        self,
        archive: NominalDemonstrationArchive,
        episode_indexes: np.ndarray,
        *,
        history_length: int = 2,
        action_horizon: int = 8,
        normalization: dict[str, list[float]] | None = None,
    ) -> None:
        if history_length < 1 or action_horizon < 1:
            raise ValueError("history_length and action_horizon must be positive")
        self.archive = archive
        self.history_length = int(history_length)
        self.action_horizon = int(action_horizon)
        self.anchors = [
            (int(episode), int(step))
            for episode in np.asarray(episode_indexes).reshape(-1)
            for step in np.flatnonzero(archive.valid[int(episode)])
        ]
        if not self.anchors:
            raise ValueError("Dataset split contains no valid timesteps")
        normalization = normalization or {
            "mean": [0.0] * archive.proprio.shape[-1],
            "std": [1.0] * archive.proprio.shape[-1],
        }
        self.proprio_mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.proprio_std = np.asarray(normalization["std"], dtype=np.float32)
        if self.proprio_mean.shape != (10,) or self.proprio_std.shape != (10,):
            raise ValueError("Proprio normalization must contain ten values")
        if not np.isfinite(self.proprio_mean).all() or not np.isfinite(self.proprio_std).all():
            raise ValueError("Proprio normalization must be finite")
        if np.any(self.proprio_std <= 0):
            raise ValueError("Proprio normalization standard deviations must be positive")

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, anchor = self.anchors[index]
        height, width = self.archive.rgb.shape[-2:]
        rgbd_history = np.zeros((self.history_length, 4, height, width), dtype=np.float32)
        proprio_history = np.zeros((self.history_length, 10), dtype=np.float32)
        observation_mask = np.zeros(self.history_length, dtype=np.bool_)
        history_start = max(0, anchor - self.history_length + 1)
        source_steps = np.arange(history_start, anchor + 1)
        destination_start = self.history_length - len(source_steps)
        rgbd_history[destination_start:, :3] = (
            self.archive.rgb[episode, source_steps].astype(np.float32) / 255.0
        )
        rgbd_history[destination_start:, 3:] = self.archive.depth[episode, source_steps].astype(
            np.float32
        )
        proprio = self.archive.proprio[episode, source_steps]
        proprio_history[destination_start:] = (proprio - self.proprio_mean) / self.proprio_std
        observation_mask[destination_start:] = True

        action_chunk = np.zeros((self.action_horizon, 4), dtype=np.float32)
        action_mask = np.zeros(self.action_horizon, dtype=np.bool_)
        episode_length = int(self.archive.valid[episode].sum())
        future_end = min(episode_length, anchor + self.action_horizon)
        count = future_end - anchor
        action_chunk[:count] = self.archive.actions[episode, anchor:future_end]
        action_mask[:count] = True
        return {
            "rgbd_history": torch.from_numpy(rgbd_history),
            "proprio_history": torch.from_numpy(proprio_history),
            "observation_mask": torch.from_numpy(observation_mask),
            "action_chunk": torch.from_numpy(action_chunk),
            "action_mask": torch.from_numpy(action_mask),
        }
