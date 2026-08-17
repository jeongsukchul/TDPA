from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("robosuite")

from tdpa.envs.base import DEPLOYMENT_KEYS, Physics
from tdpa.envs.make_env import make_env
from tdpa.policies.frozen_nominal import assert_deployment_observation
from tdpa.policies.privileged_expert import PrivilegedScriptedExpert


@pytest.mark.simulation
@pytest.mark.parametrize("task", ["push", "lift"])
def test_robosuite_reset_step_and_live_physics(task: str) -> None:
    physics = Physics(0.73, 0.27)
    env = make_env(
        task,
        physics=physics,
        seed=13,
        episode_index=2,
        backend="robosuite",
    )
    try:
        observation = env.reset()
        assert set(observation) == DEPLOYMENT_KEYS
        assert observation["rgbd"].shape == (4, env.image_size, env.image_size)
        assert observation["proprio"].shape == (10,)
        assert all(value.dtype == np.float32 for value in observation.values())
        assert all(np.isfinite(value).all() for value in observation.values())
        assert_deployment_observation(observation)

        low, high = env.action_spec
        assert low.shape == high.shape == (4,)
        before = env.read_physics()
        assert before.actual_mass == pytest.approx(physics.mass, abs=1e-9)
        assert before.object_geoms
        assert before.counterpart_geoms
        assert all(
            geom.friction[0] == pytest.approx(physics.friction, abs=1e-9)
            for geom in before.object_geoms + before.counterpart_geoms
        )

        action = env.prepare_contact_probe()
        stepped, reward, terminated, truncated, info = env.step(action)
        assert set(stepped) == DEPLOYMENT_KEYS
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert all(np.isfinite(float(value)) for value in info.values())
        after = env.read_physics()
        assert after.as_dict() == before.as_dict()
        assert env.contact_report()["relevant_contact_count"] >= 1

        env.step(
            np.zeros(4, dtype=np.float32),
            {
                "velocity_scale": 99.0,
                "stiffness": 999.0,
                "damping": float("nan"),
                "grip_force": 999.0,
            },
        )
        execution = env.controller_readback()
        configured = env.robosuite_config["execution"]
        assert execution["applied"]["velocity_scale"] == configured["bounds"]["velocity_scale"][1]
        assert execution["applied"]["stiffness"] == configured["bounds"]["stiffness"][1]
        assert execution["applied"]["damping"] == configured["nominal"]["damping"]
        assert execution["applied"]["grip_force"] == configured["bounds"]["grip_force"][1]
        assert execution["saturated"]
        assert all(
            values == [-execution["applied"]["grip_force"], execution["applied"]["grip_force"]]
            for values in execution["gripper_force_ranges"]
        )

        env.reset()
        restored = env.controller_readback()
        assert restored["applied"] == configured["nominal"]
        assert not restored["saturated"]
    finally:
        env.close()


@pytest.mark.simulation
def test_robosuite_indexed_reset_replays_and_varies() -> None:
    env = make_env("push", seed=19, episode_index=0, backend="robosuite")
    try:
        first_observation = env.reset()
        first_fingerprint = env.reset_fingerprint()
        first_robot_qpos = env.reset_state().robot_qpos
        replay_observation = env.reset()
        assert env.reset_fingerprint() == first_fingerprint
        assert all(
            np.array_equal(first_observation[key], replay_observation[key])
            for key in DEPLOYMENT_KEYS
        )

        env.episode_index = 1
        env.reset()
        assert env.reset_fingerprint() != first_fingerprint
        assert env.reset_state().robot_qpos != first_robot_qpos
    finally:
        env.close()


@pytest.mark.simulation
def test_push_success_uses_the_current_randomized_target() -> None:
    env = make_env("push", seed=23, episode_index=0, backend="robosuite")
    try:
        env.reset()
        position = np.asarray(env.raw.sim.data.body_xpos[env.raw.cube_body_id]).copy()
        tolerance = float(env.robosuite_config["success_tolerance"])
        env.raw.sim.model.site_pos[env._target_site_id] = position + np.array(
            [1.1 * tolerance, 0.0, 0.0]
        )
        env.raw.sim.forward()
        assert not env.raw._check_success()
        assert env.metrics()["final_error"] == pytest.approx(1.1 * tolerance)

        env.raw.sim.model.site_pos[env._target_site_id] = position + np.array(
            [0.9 * tolerance, 0.0, 0.0]
        )
        env.raw.sim.forward()
        assert env.raw._check_success()
        assert env.metrics()["success"]
    finally:
        env.close()


@pytest.mark.simulation
@pytest.mark.parametrize("task", ["push", "lift"])
def test_privileged_expert_can_label_a_successful_nominal_episode(task: str) -> None:
    env = make_env(task, seed=31, episode_index=0, backend="robosuite")
    expert = PrivilegedScriptedExpert(task)
    try:
        observation = env.reset()
        info = env.metrics()
        for _ in range(env.horizon):
            decision = expert.act(env, observation)
            observation, _, terminated, truncated, info = env.step(decision.action)
            if terminated or truncated:
                break
        assert info["success"]
    finally:
        env.close()
