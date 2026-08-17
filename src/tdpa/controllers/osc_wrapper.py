from __future__ import annotations

import numpy as np


class OperationalSpaceControllerWrapper:
    """Minimal safety boundary around a backend operational-space controller."""

    def __init__(self, backend: object, translation_limit: float = 0.15) -> None:
        self.backend = backend
        self.translation_limit = float(translation_limit)

    def command(self, cartesian_delta: np.ndarray, **parameters: float) -> object:
        delta = np.nan_to_num(np.asarray(cartesian_delta, dtype=np.float32))
        delta = np.clip(delta, -self.translation_limit, self.translation_limit)
        method = getattr(self.backend, "command", None)
        if method is None:
            raise TypeError("Controller backend must provide command(delta, **parameters)")
        return method(delta, **parameters)

