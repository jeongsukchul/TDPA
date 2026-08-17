from __future__ import annotations

import hashlib

import numpy as np


def apply_behavior_style(action: np.ndarray, policy_id: str, step: int) -> np.ndarray:
    """Apply a held-out, deterministic motion style without exposing its ID."""
    action = np.asarray(action, dtype=np.float32).copy()
    if policy_id == "chirp":
        phase = 0.21 * step + 0.017 * step * step
        perturbation = 0.12 * np.array(
            [np.sin(phase), np.cos(1.3 * phase), np.sin(0.7 * phase)], dtype=np.float32
        )
        action[:3] = np.clip(action[:3] + perturbation, -1.0, 1.0)
    return action


def execution_trace_hash(commands: list[np.ndarray]) -> str:
    if not commands:
        return hashlib.sha256(b"").hexdigest()
    payload = np.stack(commands).astype("<f4", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()

