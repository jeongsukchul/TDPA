from __future__ import annotations

import numpy as np
import torch

from tdpa.data.history_buffer import DeploymentHistory
from tdpa.data.sequence_dataset import FixedFormatEpisode, SequenceDataset


def test_online_history_matches_offline_strict_past_window() -> None:
    length, history_length = 7, 4
    rgbd = np.arange(length, dtype=np.float32)[:, None, None, None]
    proprio = np.stack([np.full(2, index, dtype=np.float32) for index in range(length)])
    actions = np.stack([np.full(7, index + 10, dtype=np.float32) for index in range(length)])
    responses = np.zeros((length, 2), dtype=np.float32)
    episode = FixedFormatEpisode(rgbd, proprio, actions, responses)
    dataset = SequenceDataset([episode], history_length, 2)
    online = DeploymentHistory(history_length)
    for anchor in range(1, length - 1):
        online.append(
            {"rgbd": rgbd[anchor - 1], "proprio": proprio[anchor - 1]},
            actions[anchor - 1],
        )
        tensors = online.tensors()
        offline = dataset[anchor - 1]
        assert dataset.window_alignment(anchor - 1).anchor == anchor
        for key in ("rgbd_history", "proprio_history", "action_history", "history_mask"):
            assert torch.equal(tensors[key].squeeze(0), offline[key])


def test_unpaired_future_action_cannot_change_model_input() -> None:
    episode = FixedFormatEpisode(
        np.zeros((6, 1)),
        np.zeros((6, 2)),
        np.arange(42, dtype=np.float32).reshape(6, 7),
        np.zeros((6, 2)),
    )
    dataset = SequenceDataset([episode], 2, 4)
    item = dataset[len(dataset) - 1]
    assert torch.equal(item["future_action_mask"], item["future_response_mask"])
    assert torch.count_nonzero(item["future_action_sequence"][~item["future_mask"]]) == 0
