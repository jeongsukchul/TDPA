from __future__ import annotations

import numpy as np
import pytest

from tdpa.envs.base import PRIVILEGED_KEYS, Physics
from tdpa.envs.make_env import make_env
from tdpa.models.history_encoder import HistoryEncoder
from tdpa.policies.frozen_nominal import FrozenNominalPolicy


@pytest.mark.parametrize("task", ["push", "lift"])
def test_visual_and_deployment_schema_do_not_reveal_physics(task: str) -> None:
    light = make_env(task, physics=Physics(0.6, 0.35), seed=4)
    heavy = make_env(task, physics=Physics(2.4, 1.2), seed=4)
    light_obs = light.reset()
    heavy_obs = heavy.reset()
    assert set(light_obs) == {"rgbd", "proprio"}
    assert np.array_equal(light_obs["rgbd"], heavy_obs["rgbd"])
    assert np.array_equal(light_obs["proprio"], heavy_obs["proprio"])


@pytest.mark.parametrize("forbidden", sorted(PRIVILEGED_KEYS))
def test_frozen_policy_rejects_privileged_or_evaluator_keys(forbidden: str) -> None:
    env = make_env("push")
    observation = env.reset()
    observation[forbidden] = np.zeros(1, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Privileged"):
        FrozenNominalPolicy("push", env.config)(observation)


def test_physics_encoder_masks_task_goal_render_channel() -> None:
    torch = pytest.importorskip("torch")
    encoder = HistoryEncoder(10, 7, mask_goal_channel=True).eval()
    rgbd = torch.zeros(1, 2, 4, 16, 16)
    changed = rgbd.clone()
    changed[:, :, 2, 4, 12] = 1.0
    proprio = torch.zeros(1, 2, 10)
    actions = torch.zeros(1, 2, 7)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        baseline = encoder(rgbd, proprio, actions, mask)
        goal_changed = encoder(changed, proprio, actions, mask)
    assert torch.equal(baseline, goal_changed)
