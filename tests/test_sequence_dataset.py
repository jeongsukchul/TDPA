from __future__ import annotations

import numpy as np
import pytest
import torch

from tdpa.data.sequence_dataset import FixedFormatEpisode, SequenceDataset


def make_episode(length: int = 8, *, impulse_at: int | None = None) -> FixedFormatEpisode:
    time = np.arange(length, dtype=np.float32)
    actions = np.stack((time, -time), axis=-1)
    responses = np.zeros((length, 2), dtype=np.float32)
    if impulse_at is not None:
        # action[impulse_at] causes the first physical response one tick later.
        actions[impulse_at] = (100.0, -100.0)
        responses[impulse_at + 1] = (7.0, 11.0)
    return FixedFormatEpisode(
        rgbd=time[:, None, None, None],
        proprio=np.stack((time, time + 0.5), axis=-1),
        actions=actions,
        responses=responses,
        privileged={"mass": np.full((length, 1), 42.0, dtype=np.float32)},
        metadata={"physics": {"mass": 42.0}, "split": "ood_mass"},
    )


def test_episode_rejects_non_fixed_time_dimensions() -> None:
    with pytest.raises(ValueError, match="share one time dimension"):
        FixedFormatEpisode(
            rgbd=np.zeros((5, 1)),
            proprio=np.zeros((4, 2)),
            actions=np.zeros((5, 2)),
            responses=np.zeros((5, 2)),
        )


def test_dataset_rejects_cross_episode_schema_or_length_mismatch() -> None:
    good = make_episode(8)
    bad_shape = FixedFormatEpisode(
        rgbd=np.zeros((8, 1, 1, 1)),
        proprio=np.zeros((8, 3)),
        actions=np.zeros((8, 2)),
        responses=np.zeros((8, 2)),
    )
    with pytest.raises(ValueError, match="schema"):
        SequenceDataset([good, bad_shape], 3, 2)
    with pytest.raises(ValueError, match="equal length"):
        SequenceDataset([good, make_episode(7)], 3, 2)


def test_history_is_strictly_past_and_left_padding_is_masked() -> None:
    dataset = SequenceDataset([make_episode()], history_length=4, future_horizon=3)
    first = dataset[0]  # first valid anchor is t=1
    assert dataset.window_alignment(0).anchor == 1
    assert first["history_mask"].tolist() == [False, False, False, True]
    assert first["proprio_history"][:, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert first["action_history"][-1].tolist() == [0.0, -0.0]
    assert first["future_action_sequence"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert dataset.window_alignment(0).history_indices == (-1, -1, -1, 0)


def test_synthetic_impulse_has_causal_action_response_alignment() -> None:
    impulse_at = 3
    dataset = SequenceDataset(
        [make_episode(9, impulse_at=impulse_at)], history_length=3, future_horizon=3
    )
    # Partial-window mode starts anchors at one, so item 2 is anchor t=3.
    item = dataset[2]
    alignment = dataset.window_alignment(2)
    assert alignment.anchor == impulse_at
    assert alignment.history_indices == (0, 1, 2)
    assert alignment.future_action_indices == (3, 4, 5)
    assert alignment.future_response_indices == (4, 5, 6)
    assert item["future_action_sequence"][0].tolist() == [100.0, -100.0]
    assert item["future_response_sequence"][0].tolist() == [7.0, 11.0]
    # Neither impulse action nor its response can leak into history.
    assert not torch.any(torch.abs(item["action_history"]) == 100.0)
    assert not torch.any(item["proprio_history"][:, 0] >= impulse_at)


def test_end_of_episode_padding_has_separate_and_joint_masks() -> None:
    dataset = SequenceDataset([make_episode(6)], history_length=2, future_horizon=4)
    item = dataset[len(dataset) - 1]  # last valid anchor: t=4
    assert dataset.window_alignment(len(dataset) - 1).anchor == 4
    assert item["future_action_mask"].tolist() == [True, False, False, False]
    assert item["future_response_mask"].tolist() == [True, False, False, False]
    assert item["future_mask"].tolist() == [True, False, False, False]
    assert torch.count_nonzero(item["future_action_sequence"][1:]) == 0
    assert torch.count_nonzero(item["future_response_sequence"][1:]) == 0


def test_complete_window_mode_emits_no_padding() -> None:
    dataset = SequenceDataset(
        [make_episode(8)],
        history_length=3,
        future_horizon=2,
        allow_partial_windows=False,
    )
    assert len(dataset) == 3  # anchors 3, 4, 5
    for item in dataset:
        assert item["history_mask"].all()
        assert item["future_mask"].all()


def test_privileged_data_and_metadata_are_never_emitted() -> None:
    item = SequenceDataset([make_episode()], 2, 2)[0]
    assert "privileged" not in item
    assert "metadata" not in item
    assert "mass" not in item
    assert "split" not in item
    assert 42.0 not in item["proprio_history"]


def test_repeated_reads_are_deterministic() -> None:
    episode = make_episode()
    dataset = SequenceDataset([episode], 3, 2)
    first = dataset[2]
    second = dataset[2]
    for key in first:
        assert torch.equal(first[key], second[key])


def test_input_arrays_are_copied_when_episode_is_created() -> None:
    actions = np.arange(16, dtype=np.float32).reshape(8, 2)
    episode = FixedFormatEpisode(
        rgbd=np.zeros((8, 1)),
        proprio=np.zeros((8, 2)),
        actions=actions,
        responses=np.zeros((8, 2)),
    )
    actions.fill(999.0)
    dataset = SequenceDataset([episode], 2, 2)
    assert not torch.any(dataset[0]["future_action_sequence"] == 999.0)
