from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Tiny append-only logger that remains usable without external services."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **values: Any) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")

