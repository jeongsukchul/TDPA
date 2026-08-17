from __future__ import annotations

import torch
from torch.utils.data import Dataset

from tdpa.data.sequence_dataset import SequenceDataset


class PrivilegedTeacherDataset(Dataset[dict[str, torch.Tensor]]):
    """Explicitly training-only view that adds teacher history to safe windows."""

    def __init__(self, deployment_dataset: SequenceDataset, key: str = "teacher_history") -> None:
        self.deployment_dataset = deployment_dataset
        self.key = key
        for episode in deployment_dataset.episodes:
            if key not in episode.privileged:
                raise ValueError(f"Training episode lacks privileged key {key!r}")

    def __len__(self) -> int:
        return len(self.deployment_dataset)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = dict(self.deployment_dataset[item])
        alignment = self.deployment_dataset.window_alignment(item)
        source = self.deployment_dataset.episodes[alignment.episode_index].privileged[self.key]
        history, mask = self.deployment_dataset._gather(source, alignment.history_indices)
        if not torch.equal(mask, result["history_mask"]):
            raise AssertionError("Privileged/deployment temporal alignment differs")
        result["privileged_history"] = history
        return result

