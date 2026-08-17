from __future__ import annotations

from typing import Any


class NoAdaptation:
    name = "no_adaptation"

    def correction(self, *_: object, **__: object) -> dict[str, Any]:
        return {}

