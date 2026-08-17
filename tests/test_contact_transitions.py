from __future__ import annotations

import numpy as np

from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env


def test_push_transitions_from_free_space_to_contact_and_back() -> None:
    env = make_env("push", physics=Physics(1.0, 0.55))
    env.reset()
    assert not env.contact
    observed_contact = False
    for _ in range(20):
        env.step(np.array([1.0, 0.0, -0.1, -1.0], dtype=np.float32))
        observed_contact = observed_contact or env.contact
    assert observed_contact
    for _ in range(8):
        env.step(np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert not env.contact


def test_lift_transitions_stick_to_slip_and_restick() -> None:
    env = make_env("lift", physics=Physics(1.0, 0.55))
    env.reset()
    env.ee_pos[:] = env.obj_pos
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    env.step(action, {"grip_force": 30.0, "damping": 15.0})
    assert env.contact_mode == "stick"
    env.step(action, {"grip_force": 3.0, "damping": 15.0})
    assert env.contact_mode == "slip"
    env.step(action, {"grip_force": 30.0, "damping": 15.0})
    assert env.contact_mode == "stick"

