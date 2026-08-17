from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
import torch


class DeploymentHistory:
    """Fixed-length deployment-only history with explicit left-padding mask."""

    def __init__(self, history_length: int, device: str | torch.device = "cpu") -> None:
        self.history_length = history_length
        self.device = torch.device(device)
        self.rgbd: deque[np.ndarray] = deque(maxlen=history_length)
        self.proprio: deque[np.ndarray] = deque(maxlen=history_length)
        self.actions: deque[np.ndarray] = deque(maxlen=history_length)

    def append(self, observation: Mapping[str, np.ndarray], action: np.ndarray) -> None:
        if set(observation) != {"rgbd", "proprio"}:
            raise RuntimeError("History accepts deployment observation fields only")
        self.rgbd.append(np.asarray(observation["rgbd"], dtype=np.float32).copy())
        self.proprio.append(np.asarray(observation["proprio"], dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self.actions:
            raise RuntimeError("History is empty")
        count = len(self.actions)
        pad = self.history_length - count

        def padded(values: deque[np.ndarray]) -> torch.Tensor:
            shape = (pad, *values[0].shape)
            array = np.concatenate(
                [np.zeros(shape, dtype=np.float32), np.stack(values)], axis=0
            )
            return torch.from_numpy(array).unsqueeze(0).to(self.device)

        mask = torch.zeros((1, self.history_length), dtype=torch.bool, device=self.device)
        mask[:, pad:] = True
        return {
            "rgbd_history": padded(self.rgbd),
            "proprio_history": padded(self.proprio),
            "action_history": padded(self.actions),
            "history_mask": mask,
        }

