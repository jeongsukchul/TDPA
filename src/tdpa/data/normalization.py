from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor, epsilon: float = 1e-6) -> NormalizationStats:
        if values.numel() == 0:
            raise ValueError("Cannot fit normalization to an empty tensor")
        flattened = values.reshape(-1, values.shape[-1]).float()
        return cls(flattened.mean(0), flattened.std(0, unbiased=False).clamp_min(epsilon))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device)) / self.std.to(values.device)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)

