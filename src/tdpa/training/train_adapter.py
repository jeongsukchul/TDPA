from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.controllers.oracle_context import OracleContextAdapter
from tdpa.data.history_buffer import DeploymentHistory
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import load_physics_config
from tdpa.envs.physics_randomization import PhysicsRandomizer, PhysicsSplit
from tdpa.models.bundle import DeploymentEncoderArtifact, load_deployment_encoder
from tdpa.models.physical_adapter import PhysicalAdapter
from tdpa.policies.frozen_nominal import FrozenNominalPolicy
from tdpa.utils.checkpoints import file_sha256, save_checkpoint
from tdpa.utils.config import load_yaml
from tdpa.utils.seed import seed_everything


def _target_vector(correction: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            *np.asarray(correction.get("cartesian_residual", np.zeros(3)), dtype=np.float32),
            float(correction.get("velocity_scale", 1.0)),
            float(correction.get("stiffness", 100.0)),
            float(correction.get("damping", 15.0)),
            float(correction.get("grip_force", 18.0)),
        ],
        dtype=np.float32,
    )


def _output_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            outputs["cartesian_residual"],
            outputs["velocity_scale"],
            outputs["stiffness"],
            outputs["damping"],
            outputs["grip_force"],
        ],
        dim=-1,
    )


def _loss_scale(config: dict[str, Any], device: str) -> torch.Tensor:
    bounds = config["bounds"]
    return torch.tensor(
        [
            *([float(bounds["residual_max"])] * 3),
            float(bounds["velocity_scale"][1] - bounds["velocity_scale"][0]),
            float(bounds["stiffness"][1] - bounds["stiffness"][0]),
            float(bounds["damping"][1] - bounds["damping"][0]),
            float(bounds["grip_force"][1] - bounds["grip_force"][0]),
        ],
        device=device,
    ).clamp_min(1e-6)


@torch.no_grad()
def collect_training_tensors(
    task: str,
    bundle: DeploymentEncoderArtifact,
    episode_count: int,
    seed: int,
    device: str,
) -> TensorDataset:
    encoder = bundle.freeze_deployment_encoder()
    config = bundle.config
    randomizer = PhysicsRandomizer(load_physics_config(), seed)
    mapper = AdapterActionMapper(load_yaml(f"configs/adapter/{task}.yaml"))
    oracle = OracleContextAdapter(task, enabled=True)
    nominal_values: list[torch.Tensor] = []
    latent_values: list[torch.Tensor] = []
    proprio_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    for episode_index in range(episode_count):
        sample = randomizer.sample_at(PhysicsSplit.TRAIN, episode_index)
        physics = Physics(sample.mass, sample.friction)
        env = make_env(task, physics=physics, seed=seed + episode_index)
        observation = env.reset()
        policy = FrozenNominalPolicy(task, env.config)
        history = DeploymentHistory(int(config["history_length"]), device)
        for step in range(env.horizon):
            nominal = policy(observation)
            if step == 0:
                applied = mapper.apply(nominal)
            else:
                tensors = history.tensors()
                rgbd = tensors["rgbd_history"] if config.get("use_rgbd", True) else None
                latent = encoder(
                    rgbd,
                    tensors["proprio_history"],
                    tensors["action_history"],
                    tensors["history_mask"],
                )
                correction = oracle.correction(physics)
                nominal_values.append(torch.from_numpy(nominal).view(1, -1))
                latent_values.append(latent.cpu())
                proprio_values.append(torch.from_numpy(observation["proprio"]).view(1, -1))
                target_values.append(torch.from_numpy(_target_vector(correction)).view(1, -1))
                applied = mapper.apply(nominal, correction)
            previous = observation
            observation, _, terminated, truncated, _ = env.step(applied.action, applied.controller)
            history.append(previous, applied.execution_command)
            if terminated or truncated:
                break
    return TensorDataset(
        torch.cat(nominal_values),
        torch.cat(latent_values),
        torch.cat(proprio_values),
        torch.cat(target_values),
    )


def train(
    *,
    task: str,
    encoder: Path,
    output: Path,
    episodes: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    seed_everything(seed)
    bundle = load_deployment_encoder(str(encoder), device)
    adapter_config = load_yaml(f"configs/adapter/{task}.yaml")
    adapter = PhysicalAdapter(adapter_config).to(device)
    data = collect_training_tensors(task, bundle, episodes, seed, device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4)
    scale = _loss_scale(adapter_config, device)
    final_loss = float("nan")
    adapter.train()
    for _ in range(epochs):
        total = 0.0
        steps = 0
        for nominal, latent, proprio, target in loader:
            nominal, latent, proprio, target = (
                nominal.to(device), latent.to(device), proprio.to(device), target.to(device)
            )
            prediction = _output_vector(adapter(nominal, latent, proprio))
            loss = torch.mean(torch.square((prediction - target) / scale))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        final_loss = total / max(steps, 1)
    save_checkpoint(
        output,
        model=adapter,
        config=adapter_config,
        optimizer=optimizer,
        metadata={
            "task": task,
            "seed": seed,
            "epochs": epochs,
            "task_specific_episodes": episodes,
            "task_specific_samples": len(data),
            "encoder_checkpoint": str(encoder),
            "encoder_hash": file_sha256(encoder),
            "physics_config_hashes": {
                "train": file_sha256("configs/physics/train.yaml"),
                "ood": file_sha256("configs/physics/ood.yaml"),
            },
            "base_policy_checkpoint": "synthetic_visual_servo_v1",
            "encoder_frozen": True,
            "base_policy_frozen": True,
            "trainable_task_parameters": adapter.trainable_parameter_count,
            "final_loss": final_loss,
        },
    )
    return {"loss": final_loss, "samples": float(len(data))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small adapter with a frozen encoder")
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(train(**vars(args)))


if __name__ == "__main__":
    main()
