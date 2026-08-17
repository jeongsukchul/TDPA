from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tdpa.models.history_encoder import HistoryEncoder
from tdpa.models.privileged_encoder import PrivilegedEncoder
from tdpa.models.response_encoder import ResponseEncoder
from tdpa.models.response_predictor import ResponsePredictor


class EncoderBundle(nn.Module):
    """Checkpointable V1/V2/V3 modules with a single deployment encoder."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.student = HistoryEncoder(
            proprio_dim=int(config["proprio_dim"]),
            action_dim=int(config["action_dim"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            image_channels=int(config.get("image_channels", 4)),
            use_rgbd=bool(config.get("use_rgbd", True)),
            mask_goal_channel=bool(config.get("mask_goal_channel", True)),
        )
        self.teacher = PrivilegedEncoder(
            privileged_dim=int(config["privileged_dim"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        self.response_encoder = ResponseEncoder(
            response_dim=int(config["response_dim"]),
            embedding_dim=int(config["response_embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        self.student_predictor = ResponsePredictor(
            latent_dim=int(config["latent_dim"]),
            action_dim=int(config["action_dim"]),
            embedding_dim=int(config["response_embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        self.teacher_predictor = ResponsePredictor(
            latent_dim=int(config["latent_dim"]),
            action_dim=int(config["action_dim"]),
            embedding_dim=int(config["response_embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )

    def freeze_deployment_encoder(self) -> HistoryEncoder:
        self.student.eval()
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        return self.student


class DeploymentEncoderArtifact(nn.Module):
    """Student-only artifact that cannot contain privileged teacher modules."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.student = HistoryEncoder(
            proprio_dim=int(config["proprio_dim"]),
            action_dim=int(config["action_dim"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            image_channels=int(config.get("image_channels", 4)),
            use_rgbd=bool(config.get("use_rgbd", True)),
            mask_goal_channel=bool(config.get("mask_goal_channel", True)),
        )

    def freeze_deployment_encoder(self) -> HistoryEncoder:
        self.student.eval()
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        return self.student


def load_encoder_bundle(path: str, device: torch.device | str = "cpu") -> EncoderBundle:
    payload = torch.load(path, map_location=device, weights_only=False)
    bundle = EncoderBundle(payload["config"])
    bundle.load_state_dict(payload["model"])
    return bundle.to(device)


def load_deployment_encoder(
    path: str, device: torch.device | str = "cpu"
) -> DeploymentEncoderArtifact:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("metadata", {}).get("artifact_type") != "deployment_student":
        raise RuntimeError(
            "Deployment requires a student-only checkpoint (suffix '_student.pt'), not a training bundle"
        )
    artifact = DeploymentEncoderArtifact(payload["config"])
    artifact.student.load_state_dict(payload["model"])
    return artifact.to(device)
