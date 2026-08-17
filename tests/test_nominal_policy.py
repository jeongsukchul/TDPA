from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tdpa.models.nominal_bc import VisualActionChunkBC
from tdpa.policies.learned_nominal import (
    FrozenLearnedNominalPolicy,
    checkpoint_sha256,
    save_nominal_checkpoint,
)
from tdpa.training.train_nominal_policy import _masked_action_loss


def test_visual_action_chunk_shape_bounds_and_mask_contract() -> None:
    model = VisualActionChunkBC(history_length=2, action_horizon=8)
    rgbd = torch.rand(3, 2, 4, 64, 64)
    proprio = torch.rand(3, 2, 10)
    mask = torch.tensor([[False, True], [True, True], [True, True]])
    output = model(rgbd, proprio, mask)
    assert output.shape == (3, 8, 4)
    assert torch.isfinite(output).all()
    assert torch.max(torch.abs(output)) <= 1.0
    with pytest.raises(ValueError, match="most recent"):
        model(rgbd, proprio, torch.zeros_like(mask))


def test_spatial_encoder_retains_checkpointed_geometry_contract() -> None:
    model = VisualActionChunkBC(
        history_length=2,
        action_horizon=8,
        vision_encoder="spatial",
    )
    output = model(
        torch.rand(2, 2, 4, 64, 64),
        torch.rand(2, 2, 10),
        torch.ones(2, 2, dtype=torch.bool),
    )
    assert output.shape == (2, 8, 4)
    assert model.model_config()["vision_encoder"] == "spatial"


def test_masked_loss_ignores_padded_actions() -> None:
    prediction = torch.zeros(1, 3, 4)
    target = torch.zeros_like(prediction)
    target[:, 1:] = 1000
    mask = torch.tensor([[True, False, False]])
    assert _masked_action_loss(prediction, target, mask) == 0


def test_untrained_checkpoint_is_rejected_by_default_and_frozen_when_explicit(
    tmp_path,
) -> None:
    model = VisualActionChunkBC()
    path = tmp_path / "smoke.pt"
    normalization = {"mean": [0.0] * 10, "std": [1.0] * 10}
    save_nominal_checkpoint(
        path,
        model=model,
        task="push",
        normalization=normalization,
        status="untrained_smoke",
        provenance={"training_steps": 0, "eligible_for_results": False},
    )
    assert path.with_suffix(".manifest.json").is_file()
    with pytest.raises(RuntimeError, match="rejected by default"):
        FrozenLearnedNominalPolicy(path, task="push")
    policy = FrozenLearnedNominalPolicy(path, task="push", allow_untrained=True)
    assert policy.frozen
    observation = {
        "rgbd": np.zeros((4, 64, 64), dtype=np.float32),
        "proprio": np.zeros(10, dtype=np.float32),
    }
    action = policy(observation)
    assert action.shape == (4,)
    assert np.isfinite(action).all()
    assert action[3] in {-1.0, 1.0}
    with pytest.raises(ValueError, match="shape"):
        policy(
            {
                "rgbd": np.zeros((4, 32, 32), dtype=np.float32),
                "proprio": np.zeros(10, dtype=np.float32),
            }
        )
    out_of_range = observation["rgbd"].copy()
    out_of_range[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="range"):
        policy({"rgbd": out_of_range, "proprio": observation["proprio"]})
    with pytest.raises(TypeError, match="float32"):
        policy(
            {
                "rgbd": observation["rgbd"].astype(np.float64),
                "proprio": observation["proprio"],
            }
        )
    with pytest.raises(ValueError, match="does not match"):
        FrozenLearnedNominalPolicy(path, task="lift", allow_untrained=True)


def test_trained_checkpoint_requires_complete_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="provenance is missing"):
        save_nominal_checkpoint(
            tmp_path / "invalid.pt",
            model=VisualActionChunkBC(),
            task="push",
            normalization={"mean": [0.0] * 10, "std": [1.0] * 10},
            status="trained",
            provenance={"training_steps": 1},
        )


def test_torch_version_provenance_is_normalized_and_legacy_checkpoint_loads(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    save_nominal_checkpoint(
        path,
        model=VisualActionChunkBC(),
        task="push",
        normalization={"mean": [0.0] * 10, "std": [1.0] * 10},
        status="untrained_smoke",
        provenance={
            "training_steps": 0,
            "eligible_for_results": False,
            "torch": torch.__version__,
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert type(payload["provenance"]["torch"]) is str

    payload["provenance"]["torch"] = torch.__version__
    torch.save(payload, path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = checkpoint_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    policy = FrozenLearnedNominalPolicy(path, task="push", allow_untrained=True)
    assert policy.frozen


def test_legacy_global_checkpoint_without_encoder_key_still_loads(tmp_path) -> None:
    path = tmp_path / "legacy_global.pt"
    save_nominal_checkpoint(
        path,
        model=VisualActionChunkBC(vision_encoder="global"),
        task="push",
        normalization={"mean": [0.0] * 10, "std": [1.0] * 10},
        status="untrained_smoke",
        provenance={"training_steps": 0, "eligible_for_results": False},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["model_config"].pop("vision_encoder")
    torch.save(payload, path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = checkpoint_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    policy = FrozenLearnedNominalPolicy(path, task="push", allow_untrained=True)
    assert policy.model.vision_encoder == "global"
