"""Strict frozen loader for learned task-specific nominal policies."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from tdpa.models.nominal_bc import VisualActionChunkBC
from tdpa.policies.frozen_nominal import assert_deployment_observation
from tdpa.utils.config import load_yaml

CHECKPOINT_VERSION = "tdpa-nominal-policy-v1"
ALLOWED_STATUSES = {"trained", "untrained_smoke"}
_TRAINED_PROVENANCE_FIELDS = {
    "training_steps",
    "epochs",
    "dataset_sha256",
    "split_sha256",
    "train_episode_ids",
    "validation_episode_ids",
    "collection_seed",
    "attempted_episodes",
    "successful_episodes",
    "competence_gate",
}


@runtime_checkable
class NominalPolicy(Protocol):
    task: str

    @property
    def frozen(self) -> bool: ...

    def reset(self) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray: ...


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_environment_hash(task: str) -> str:
    return stable_hash(load_yaml(f"configs/env/{task}.yaml"))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_specs(task: str) -> tuple[dict[str, object], dict[str, object]]:
    robosuite_config = load_yaml(f"configs/env/{task}.yaml")["robosuite"]
    size = int(robosuite_config["image_size"])
    observation_spec: dict[str, object] = {
        "keys": ["rgbd", "proprio"],
        "rgbd_shape": [4, size, size],
        "proprio_shape": [10],
        "dtype": "float32",
        "rgbd_range": [0.0, 1.0],
    }
    action_spec: dict[str, object] = {
        "order": ["delta_x", "delta_y", "delta_z", "gripper"],
        "shape": [4],
        "normalized_bounds": [-1.0, 1.0],
        "position_delta_limit": float(robosuite_config["position_delta_limit"]),
        "gripper_convention": "-1=open,+1=close",
    }
    return observation_spec, action_spec


def _validate_provenance(status: str, provenance: Mapping[str, Any]) -> None:
    if status == "untrained_smoke":
        if (
            provenance.get("training_steps") != 0
            or provenance.get("eligible_for_results") is not False
        ):
            raise ValueError("Smoke provenance must declare zero steps and result ineligibility")
        return
    missing = _TRAINED_PROVENANCE_FIELDS - set(provenance)
    if missing:
        raise ValueError(f"Trained checkpoint provenance is missing: {sorted(missing)}")
    if not isinstance(provenance["training_steps"], int) or provenance["training_steps"] <= 0:
        raise ValueError("Trained checkpoint must report positive optimizer steps")
    if not _is_sha256(provenance["dataset_sha256"]) or not _is_sha256(provenance["split_sha256"]):
        raise ValueError("Trained checkpoint dataset/split hashes are invalid")
    train_ids = set(provenance["train_episode_ids"])
    validation_ids = set(provenance["validation_episode_ids"])
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("Trained checkpoint episode splits must be non-empty and disjoint")
    if not isinstance(provenance["collection_seed"], int):
        raise TypeError("Trained checkpoint collection seed is invalid")
    if provenance["competence_gate"] != "not_run":
        raise ValueError("Newly trained checkpoint must not claim an embedded competence result")


def save_nominal_checkpoint(
    path: Path,
    *,
    model: VisualActionChunkBC,
    task: str,
    normalization: Mapping[str, list[float]],
    status: str,
    provenance: Mapping[str, Any],
) -> None:
    if task not in {"push", "lift"}:
        raise ValueError("Checkpoint task must be push or lift")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unknown checkpoint status: {status}")
    try:
        normalized_provenance = json.loads(json.dumps(dict(provenance), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError("Checkpoint provenance must contain only JSON-safe values") from error
    if not isinstance(normalized_provenance, dict):
        raise TypeError("Checkpoint provenance must normalize to a mapping")
    _validate_provenance(status, normalized_provenance)
    mean = torch.as_tensor(normalization["mean"], dtype=torch.float32)
    std = torch.as_tensor(normalization["std"], dtype=torch.float32)
    if mean.shape != (model.proprio_dim,) or std.shape != (model.proprio_dim,):
        raise ValueError("Checkpoint normalization shape mismatch")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Checkpoint normalization must be finite with positive std")
    observation_spec, action_spec = _expected_specs(task)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_family": model.family,
        "model_config": model.model_config(),
        "task": task,
        "backend": "robosuite",
        "frozen": True,
        "status": status,
        "environment_hash": current_environment_hash(task),
        "observation_spec": observation_spec,
        "action_spec": action_spec,
        "context_horizon": model.history_length,
        "prediction_horizon": model.action_horizon,
        "execution_horizon": 1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "normalization_mean": mean.cpu(),
        "normalization_std": std.cpu(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "provenance": normalized_provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "normalization_mean", "normalization_std"}
    }
    manifest["normalization"] = {"mean": mean.tolist(), "std": std.tolist()}
    manifest["checkpoint_sha256"] = checkpoint_sha256(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class FrozenLearnedNominalPolicy:
    """Stateful deployment wrapper that never exposes privileged checkpoint data."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        task: str,
        device: str | torch.device = "cpu",
        allow_untrained: bool = False,
        expected_environment_hash: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        manifest_path = self.checkpoint_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError("Nominal checkpoint sidecar manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_sha256") != checkpoint_sha256(self.checkpoint_path):
            raise ValueError("Nominal checkpoint hash does not match its sidecar manifest")
        # Checkpoints written before the provenance normalizer stored
        # ``torch.__version__`` as this benign ``str`` subclass. Keep the
        # restricted weights-only loader and narrowly allowlist that one type.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("Nominal checkpoint must contain a mapping")
        required = {
            "checkpoint_version",
            "model_family",
            "model_config",
            "task",
            "backend",
            "frozen",
            "status",
            "environment_hash",
            "observation_spec",
            "action_spec",
            "context_horizon",
            "prediction_horizon",
            "execution_horizon",
            "parameter_count",
            "normalization_mean",
            "normalization_std",
            "model_state",
            "provenance",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Nominal checkpoint is missing fields: {sorted(missing)}")
        if payload["checkpoint_version"] != CHECKPOINT_VERSION:
            raise ValueError("Unsupported nominal checkpoint version")
        if payload["model_family"] != VisualActionChunkBC.family:
            raise ValueError("Unsupported nominal model family")
        if payload["backend"] != "robosuite" or payload["frozen"] is not True:
            raise ValueError("Nominal checkpoint must declare a frozen robosuite policy")
        if payload["task"] != task:
            raise ValueError(f"Checkpoint task {payload['task']!r} does not match {task!r}")
        if payload["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid checkpoint status: {payload['status']!r}")
        if payload["status"] != "trained" and not allow_untrained:
            raise RuntimeError(
                "Untrained smoke checkpoints are rejected by default; pass allow_untrained=True "
                "only in interface smoke tests"
            )
        expected_hash = expected_environment_hash or current_environment_hash(task)
        if payload["environment_hash"] != expected_hash:
            raise ValueError("Checkpoint environment hash does not match the current task config")
        _validate_provenance(str(payload["status"]), payload["provenance"])

        config = dict(payload["model_config"])
        if payload["context_horizon"] != config.get("history_length"):
            raise ValueError("Checkpoint context horizon is inconsistent")
        if payload["prediction_horizon"] != config.get("action_horizon"):
            raise ValueError("Checkpoint prediction horizon is inconsistent")
        if payload["execution_horizon"] != 1:
            raise ValueError("Only receding-horizon one-step execution is supported")
        expected_observation_spec, expected_action_spec = _expected_specs(task)
        if payload["observation_spec"] != expected_observation_spec:
            raise ValueError("Checkpoint observation schema is incompatible")
        if payload["action_spec"] != expected_action_spec:
            raise ValueError("Checkpoint action schema is incompatible")
        if config.get("proprio_dim") != 10 or config.get("action_dim") != 4:
            raise ValueError("Checkpoint model dimensions are incompatible")
        self.device = torch.device(device)
        self.model = VisualActionChunkBC(**config).to(self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)
        actual_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        if payload["parameter_count"] != actual_parameters:
            raise ValueError("Checkpoint parameter count is inconsistent")
        self.model.eval()
        self.model.requires_grad_(False)
        self.task = task
        self.status = str(payload["status"])
        self.provenance = dict(payload["provenance"])
        self.observation_spec = dict(payload["observation_spec"])
        self.proprio_mean = torch.as_tensor(
            payload["normalization_mean"], dtype=torch.float32, device=self.device
        )
        self.proprio_std = torch.as_tensor(
            payload["normalization_std"], dtype=torch.float32, device=self.device
        )
        if self.proprio_mean.shape != (self.model.proprio_dim,):
            raise ValueError("Checkpoint normalization dimension mismatch")
        if (
            not torch.isfinite(self.proprio_mean).all()
            or not torch.isfinite(self.proprio_std).all()
        ):
            raise ValueError("Checkpoint normalization is non-finite")
        if torch.any(self.proprio_std <= 0):
            raise ValueError("Checkpoint normalization std must be positive")
        self._rgbd: deque[np.ndarray] = deque(maxlen=self.model.history_length)
        self._proprio: deque[np.ndarray] = deque(maxlen=self.model.history_length)

    @property
    def frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.model.parameters())

    def reset(self) -> None:
        self._rgbd.clear()
        self._proprio.clear()

    def _history(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = self.model.history_length
        height, width = self._rgbd[-1].shape[-2:]
        rgbd = np.zeros((history, 4, height, width), dtype=np.float32)
        proprio = np.zeros((history, self.model.proprio_dim), dtype=np.float32)
        mask = np.zeros(history, dtype=np.bool_)
        start = history - len(self._rgbd)
        rgbd[start:] = np.stack(self._rgbd)
        proprio[start:] = np.stack(self._proprio)
        mask[start:] = True
        rgbd_tensor = torch.from_numpy(rgbd).unsqueeze(0).to(self.device)
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        proprio_tensor = (proprio_tensor - self.proprio_mean) / self.proprio_std
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return rgbd_tensor, proprio_tensor, mask_tensor

    @torch.inference_mode()
    def predict_action_chunk(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        assert_deployment_observation(observation)
        rgbd = np.asarray(observation["rgbd"])
        proprio = np.asarray(observation["proprio"])
        if rgbd.dtype != np.float32 or proprio.dtype != np.float32:
            raise TypeError("Deployment observations must use checkpointed float32 dtype")
        expected_rgbd = tuple(int(value) for value in self.observation_spec["rgbd_shape"])
        if rgbd.shape != expected_rgbd or proprio.shape != (10,):
            raise ValueError("Unexpected deployment observation shape")
        if not np.isfinite(rgbd).all() or not np.isfinite(proprio).all():
            raise ValueError("Deployment observation must be finite")
        rgbd_low, rgbd_high = self.observation_spec["rgbd_range"]
        if np.any(rgbd < rgbd_low) or np.any(rgbd > rgbd_high):
            raise ValueError("RGB-D observation is outside its checkpointed range")
        self._rgbd.append(rgbd.copy())
        self._proprio.append(proprio.copy())
        inputs = self._history()
        chunk = self.model.predict_action_chunk(*inputs)[0].cpu().numpy().astype(np.float32)
        if chunk.shape != (self.model.action_horizon, self.model.action_dim):
            raise RuntimeError("Nominal policy emitted an invalid action chunk")
        if not np.isfinite(chunk).all():
            raise RuntimeError("Nominal policy emitted non-finite actions")
        return np.clip(chunk, -1.0, 1.0)

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        action = self.predict_action_chunk(observation)[0].copy()
        action[3] = 1.0 if action[3] >= 0.0 else -1.0
        return action

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.act(observation)
