from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def force_summary(forces: Iterable[float], force_limit: float) -> dict[str, float]:
    values = np.asarray(list(forces), dtype=np.float64)
    if values.size == 0:
        return {"peak_contact_force": 0.0, "rms_contact_force": 0.0, "force_violation_rate": 0.0}
    return {
        "peak_contact_force": float(values.max()),
        "rms_contact_force": float(np.sqrt(np.mean(np.square(values)))),
        "force_violation_rate": float(np.mean(values > force_limit)),
    }


def aggregate_episode_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot aggregate an empty result set")
    fields = {
        "success": "success_rate",
        "final_error": "final_error",
        "completion_time": "completion_time",
        "peak_contact_force": "peak_contact_force",
        "rms_contact_force": "rms_contact_force",
        "force_violation_rate": "force_violation_rate",
        "drop": "drop_rate",
        "slip": "slip_rate",
        "saturation_rate": "saturation_rate",
    }
    result: dict[str, float] = {"episodes": float(len(rows))}
    for source, target in fields.items():
        present = [float(row[source]) for row in rows if source in row]
        if present:
            result[target] = float(np.mean(present))
    return result

