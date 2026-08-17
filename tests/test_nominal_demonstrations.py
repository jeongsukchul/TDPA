from __future__ import annotations

import numpy as np
import torch

from tdpa.data.nominal_demonstrations import (
    SCHEMA_VERSION,
    NominalActionChunkDataset,
    NominalDemonstrationArchive,
    episode_split,
    fit_proprio_normalization,
    split_manifest_sha256,
)


def make_archive() -> NominalDemonstrationArchive:
    episodes, horizon, size = 4, 5, 8
    rgb = np.zeros((episodes, horizon, 3, size, size), dtype=np.uint8)
    depth = np.zeros((episodes, horizon, 1, size, size), dtype=np.float16)
    proprio = np.zeros((episodes, horizon, 10), dtype=np.float32)
    actions = np.zeros((episodes, horizon, 4), dtype=np.float32)
    for episode in range(episodes):
        for step in range(horizon):
            rgb[episode, step] = episode * 10 + step
            depth[episode, step] = (episode * 10 + step) / 100
            proprio[episode, step] = episode * 100 + step
            actions[episode, step] = [episode / 4, step / 4, -step / 4, 1]
    valid = np.ones((episodes, horizon), dtype=np.bool_)
    valid[3, 3:] = False
    terminals = np.zeros_like(valid)
    terminals[:3, -1] = True
    terminals[3, 2] = True
    return NominalDemonstrationArchive(
        rgb=rgb,
        depth=depth,
        proprio=proprio,
        actions=actions,
        valid=valid,
        terminals=terminals,
        success=np.ones(episodes, dtype=np.bool_),
        eligible=np.ones(episodes, dtype=np.bool_),
        episode_ids=np.arange(100, 104, dtype=np.int64),
        metadata={"schema_version": SCHEMA_VERSION, "task": "push"},
    )


def test_archive_round_trip_without_pickle(tmp_path) -> None:
    original = make_archive()
    for suffix in ("npz", "hdf5"):
        path = tmp_path / f"demos.{suffix}"
        original.save(path)
        loaded = NominalDemonstrationArchive.load(path)
        for name in (
            "rgb",
            "depth",
            "proprio",
            "actions",
            "valid",
            "terminals",
            "success",
            "eligible",
            "episode_ids",
        ):
            assert np.array_equal(getattr(original, name), getattr(loaded, name))
        assert loaded.metadata == original.metadata


def test_episode_split_is_disjoint_deterministic_and_hashed() -> None:
    archive = make_archive()
    first = episode_split(archive, validation_fraction=0.25, seed=9)
    second = episode_split(archive, validation_fraction=0.25, seed=9)
    assert set(first["train"]).isdisjoint(first["validation"])
    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["validation"], second["validation"])
    assert split_manifest_sha256(archive, first) == split_manifest_sha256(archive, second)


def test_normalization_uses_only_selected_episode_timesteps() -> None:
    archive = make_archive()
    normalization = fit_proprio_normalization(archive, np.array([0, 1]))
    expected = archive.proprio[:2][archive.valid[:2]]
    assert np.allclose(normalization["mean"], expected.mean(axis=0))
    assert max(normalization["mean"]) < 100


def test_action_chunks_are_causal_masked_and_do_not_cross_episodes() -> None:
    archive = make_archive()
    normalization = {"mean": [0.0] * 10, "std": [1.0] * 10}
    dataset = NominalActionChunkDataset(
        archive,
        np.array([0]),
        history_length=2,
        action_horizon=4,
        normalization=normalization,
    )
    first = dataset[0]
    assert first["observation_mask"].tolist() == [False, True]
    assert first["action_chunk"][:, 1].tolist() == [0.0, 0.25, 0.5, 0.75]
    last = dataset[len(dataset) - 1]
    assert last["proprio_history"][:, 0].tolist() == [3.0, 4.0]
    assert last["action_mask"].tolist() == [True, False, False, False]
    assert torch.count_nonzero(last["action_chunk"][1:]) == 0


def test_model_batches_expose_no_archive_identity_or_privileged_metadata() -> None:
    archive = make_archive()
    item = NominalActionChunkDataset(archive, np.array([0]), normalization=None)[0]
    assert set(item) == {
        "rgbd_history",
        "proprio_history",
        "observation_mask",
        "action_chunk",
        "action_mask",
    }
    assert not ({"episode_id", "task", "physics", "metadata"} & set(item))
