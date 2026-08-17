from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImpedanceCommand:
    cartesian_delta: np.ndarray
    velocity_scale: float
    stiffness: float
    damping: float
    grip_force: float


class ImpedanceInterface:
    """Backend-neutral command contract shared by simulation and future robots."""

    def send(self, command: ImpedanceCommand) -> None:
        raise NotImplementedError

