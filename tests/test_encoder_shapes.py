from __future__ import annotations

import pytest
import torch

from tdpa.models.bundle import EncoderBundle, load_deployment_encoder
from tdpa.models.physical_adapter import PhysicalAdapter
from tdpa.utils.checkpoints import save_checkpoint
from tdpa.utils.config import load_yaml


@pytest.mark.parametrize("variant", ["response", "distill", "hybrid"])
def test_encoder_interfaces_and_action_conditioning(variant: str) -> None:
    config = load_yaml(f"configs/encoder/{variant}.yaml")
    model = EncoderBundle(config)
    batch, history, future = 3, config["history_length"], config["future_length"]
    rgbd = torch.randn(batch, history, 4, 16, 16)
    proprio = torch.randn(batch, history, 10)
    action = torch.randn(batch, history, config["action_dim"])
    mask = torch.ones(batch, history, dtype=torch.bool)
    latent = model.student(rgbd, proprio, action, mask)
    assert latent.shape == (batch, config["latent_dim"])
    zeros = torch.zeros(batch, future, config["action_dim"])
    ones = torch.ones(batch, future, config["action_dim"])
    prediction_a = model.student_predictor(latent, zeros)
    prediction_b = model.student_predictor(latent, ones)
    assert prediction_a.shape == (batch, config["response_embedding_dim"])
    assert not torch.allclose(prediction_a, prediction_b)


def test_adapter_has_stable_dictionary_contract() -> None:
    config = load_yaml("configs/adapter/push.yaml")
    adapter = PhysicalAdapter(config)
    result = adapter(torch.zeros(2, 4), torch.zeros(2, 32), torch.zeros(2, 10))
    assert set(result) == {
        "cartesian_residual",
        "velocity_scale",
        "stiffness",
        "damping",
        "grip_force",
    }
    assert torch.all(result["damping"] == sum(config["bounds"]["damping"]) / 2)


def test_unpaired_actions_cannot_change_response_prediction() -> None:
    config = load_yaml("configs/encoder/response.yaml")
    model = EncoderBundle(config).eval()
    latent = torch.randn(2, config["latent_dim"])
    actions = torch.randn(2, config["future_length"], config["action_dim"])
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    perturbed = actions.clone()
    perturbed[~mask] = 1e6
    with torch.no_grad():
        baseline = model.student_predictor(latent, actions, mask)
        changed = model.student_predictor(latent, perturbed, mask)
    assert torch.equal(baseline, changed)


def test_deployment_loader_rejects_training_bundle(tmp_path) -> None:
    config = load_yaml("configs/encoder/response.yaml")
    bundle = EncoderBundle(config)
    training_path = tmp_path / "training.pt"
    save_checkpoint(
        training_path,
        model=bundle,
        config=config,
        metadata={"privileged_inputs": ["teacher_history"]},
    )
    with pytest.raises(RuntimeError, match="student-only"):
        load_deployment_encoder(str(training_path))


def test_response_target_changes_with_physical_response() -> None:
    config = load_yaml("configs/encoder/response.yaml")
    bundle = EncoderBundle(config).eval()
    quiet = torch.zeros(2, config["future_length"], config["response_dim"])
    impulse = quiet.clone()
    impulse[:, 0, 8:] = 5.0
    mask = torch.ones(2, config["future_length"], dtype=torch.bool)
    with torch.no_grad():
        quiet_target = bundle.response_encoder(quiet, mask)
        impulse_target = bundle.response_encoder(impulse, mask)
    assert not torch.allclose(quiet_target, impulse_target)


def test_left_padding_does_not_change_history_latent() -> None:
    encoder = EncoderBundle(load_yaml("configs/encoder/response.yaml")).student.eval()
    valid_rgbd = torch.randn(1, 2, 4, 16, 16)
    valid_proprio = torch.randn(1, 2, 10)
    valid_action = torch.randn(1, 2, 7)
    short_mask = torch.ones(1, 2, dtype=torch.bool)
    long_rgbd = torch.cat([torch.zeros(1, 3, 4, 16, 16), valid_rgbd], dim=1)
    long_proprio = torch.cat([torch.zeros(1, 3, 10), valid_proprio], dim=1)
    long_action = torch.cat([torch.zeros(1, 3, 7), valid_action], dim=1)
    long_mask = torch.tensor([[False, False, False, True, True]])
    with torch.no_grad():
        short = encoder(valid_rgbd, valid_proprio, valid_action, short_mask)
        long = encoder(long_rgbd, long_proprio, long_action, long_mask)
    assert torch.allclose(short, long, atol=1e-6)
