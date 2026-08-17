from __future__ import annotations

import torch

from tdpa.data.interaction_collector import (
    collect_interactions,
    load_episode_archive,
    save_episode_archive,
)
from tdpa.data.sequence_dataset import SequenceDataset


def test_archive_roundtrip_and_model_facing_leakage_guard(tmp_path) -> None:
    episodes = collect_interactions("push", 3, episode_length=12, seed=5)
    path = tmp_path / "interactions.npz"
    save_episode_archive(path, episodes)
    loaded = load_episode_archive(path)
    assert len(loaded) == 3
    assert torch.equal(episodes[1].responses, loaded[1].responses)
    sample = SequenceDataset(loaded, history_length=4, future_horizon=3)[2]
    assert not {"privileged", "metadata", "physics", "probe_id", "policy_id"} & set(sample)


def test_collector_records_causal_nonzero_response_after_action() -> None:
    episode = collect_interactions("push", 2, episode_length=20, seed=0)[1]
    assert torch.count_nonzero(episode.responses[0]) == 0
    assert torch.count_nonzero(episode.actions[:-1]) > 0
    assert torch.count_nonzero(episode.responses[1:]) > 0
    assert episode.actions.shape[-1] == 7
    assert torch.count_nonzero(episode.actions[-1]) == 0


def test_probe_commands_do_not_deterministically_encode_task() -> None:
    push = collect_interactions("push", 9, episode_length=16, seed=2)
    lift = collect_interactions("lift", 9, episode_length=16, seed=2)
    for push_episode, lift_episode in zip(push, lift):
        assert torch.equal(push_episode.actions, lift_episode.actions)
