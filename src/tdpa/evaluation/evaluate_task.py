from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.data.history_buffer import DeploymentHistory
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.evaluation.metrics import force_summary
from tdpa.evaluation.oracle_gate import EpisodeManifest
from tdpa.models.bundle import DeploymentEncoderArtifact
from tdpa.models.physical_adapter import PhysicalAdapter
from tdpa.policies.behavior import apply_behavior_style, execution_trace_hash
from tdpa.policies.frozen_nominal import FrozenNominalPolicy
from tdpa.utils.config import load_yaml


def load_adapter(path: str | Path, device: str = "cpu") -> PhysicalAdapter:
    payload = torch.load(path, map_location=device, weights_only=False)
    adapter = PhysicalAdapter(payload["config"])
    adapter.load_state_dict(payload["model"])
    return adapter.to(device).eval()


@torch.no_grad()
def run_adapted_episode(
    row: EpisodeManifest,
    bundle: DeploymentEncoderArtifact,
    adapter: PhysicalAdapter,
    device: str = "cpu",
) -> dict[str, Any]:
    physics = Physics(row.mass, row.friction)
    env = make_env(row.task, physics=physics, seed=row.seed * 100_000 + row.index)
    observation = env.reset()
    policy = FrozenNominalPolicy(row.task, env.config)
    mapper = AdapterActionMapper(load_yaml(f"configs/adapter/{row.task}.yaml"))
    encoder = bundle.freeze_deployment_encoder()
    history = DeploymentHistory(int(bundle.config["history_length"]), device)
    forces: list[float] = []
    saturations: list[bool] = []
    command_trace: list[np.ndarray] = []
    info: dict[str, Any] = {}
    for step in range(env.horizon):
        nominal = apply_behavior_style(policy(observation), row.behavior_policy_id, step)
        if step == 0:
            applied = mapper.apply(nominal)
        else:
            tensors = history.tensors()
            rgbd = tensors["rgbd_history"] if bundle.config.get("use_rgbd", True) else None
            latent = encoder(
                rgbd,
                tensors["proprio_history"],
                tensors["action_history"],
                tensors["history_mask"],
            )
            nominal_tensor = torch.from_numpy(nominal).to(device).view(1, -1)
            proprio = torch.from_numpy(observation["proprio"]).to(device).view(1, -1)
            correction = adapter(nominal_tensor, latent, proprio)
            applied = mapper.apply(nominal, correction)
        previous = observation
        observation, _, terminated, truncated, info = env.step(applied.action, applied.controller)
        history.append(previous, applied.execution_command)
        command_trace.append(applied.execution_command)
        forces.append(float(info["contact_force"]))
        saturations.append(applied.saturated)
        if terminated or truncated:
            break
    return {
        "task": row.task,
        "split": row.split,
        "seed": row.seed,
        "index": row.index,
        "behavior_policy_id": row.behavior_policy_id,
        "action_trace_hash": execution_trace_hash(command_trace),
        "method": "pretrained_shared",
        **{key: value for key, value in info.items() if key != "contact_force"},
        **force_summary(forces, env.force_limit),
        "saturation_rate": float(np.mean(saturations)) if saturations else 0.0,
    }
