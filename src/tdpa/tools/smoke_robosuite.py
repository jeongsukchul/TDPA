from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tdpa.envs.base import DEPLOYMENT_KEYS, Physics
from tdpa.envs.make_env import make_env
from tdpa.utils.config import load_yaml

_OBSERVATION_TOLERANCES = {"rgbd": 2.0 / 255.0 + 1e-7, "proprio": 1e-7}


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_observation(observation: dict[str, np.ndarray], image_size: int) -> None:
    if set(observation) != DEPLOYMENT_KEYS:
        raise RuntimeError(f"Deployment keys changed: {sorted(observation)}")
    if observation["rgbd"].shape != (4, image_size, image_size):
        raise RuntimeError(f"Unexpected RGB-D shape: {observation['rgbd'].shape}")
    if observation["proprio"].shape != (10,):
        raise RuntimeError(f"Unexpected proprio shape: {observation['proprio'].shape}")
    if observation["rgbd"].dtype != np.float32 or observation["proprio"].dtype != np.float32:
        raise RuntimeError("Deployment observations must be float32")
    if not all(np.isfinite(value).all() for value in observation.values()):
        raise RuntimeError("Deployment observations must be finite")


def _observation_delta(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)), initial=0.0)
    )


def _check_readback(readback: Any, physics: Physics) -> None:
    if not np.isclose(readback.actual_mass, physics.mass, rtol=0.0, atol=1e-9):
        raise RuntimeError("MuJoCo body mass readback mismatch")
    geoms = readback.object_geoms + readback.counterpart_geoms
    if not geoms:
        raise RuntimeError("No contact geoms were resolved")
    for geom in geoms:
        if not np.isclose(geom.friction[0], physics.friction, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"MuJoCo sliding friction mismatch on {geom.name}")
        if not np.isfinite(geom.friction).all():
            raise RuntimeError(f"Non-finite geom friction on {geom.name}")


def smoke(tasks: list[str], *, seed: int = 7, steps: int = 10) -> dict[str, object]:
    physics_values = (Physics(1.0, 0.55), Physics(0.4, 0.15))
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    fingerprints: dict[tuple[str, int], str] = {}
    observations: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    topology: dict[str, str] = {}
    versions: dict[str, str] = {}

    for task in tasks:
        config = load_yaml(f"configs/env/{task}.yaml")
        for physics_index, physics in enumerate(physics_values):
            env = None
            try:
                env = make_env(task, physics=physics, seed=seed, backend="robosuite")
                versions = env.versions()  # type: ignore[attr-defined]
                low, high = env.action_spec  # type: ignore[attr-defined]
                if low.shape != (4,) or high.shape != (4,):
                    raise RuntimeError("Robosuite action specification is not four-dimensional")
                for episode_index in (0, 1):
                    env.episode_index = episode_index  # type: ignore[attr-defined]
                    initial = env.reset()
                    _check_observation(initial, env.image_size)
                    fingerprint = env.reset_fingerprint()  # type: ignore[attr-defined]
                    state = env.reset_state()  # type: ignore[attr-defined]
                    key = (task, episode_index)
                    if key in fingerprints and fingerprints[key] != fingerprint:
                        raise RuntimeError("Paired reset fingerprint changed with physics")
                    fingerprints[key] = fingerprint
                    paired_observation_deltas = {obs_key: 0.0 for obs_key in DEPLOYMENT_KEYS}
                    if key in observations:
                        for obs_key in DEPLOYMENT_KEYS:
                            delta = _observation_delta(
                                observations[key][obs_key], initial[obs_key]
                            )
                            paired_observation_deltas[obs_key] = delta
                            if delta > _OBSERVATION_TOLERANCES[obs_key]:
                                raise RuntimeError(
                                    f"Initial {obs_key} revealed physics under a paired reset"
                                )
                    else:
                        observations[key] = {
                            obs_key: initial[obs_key].copy() for obs_key in DEPLOYMENT_KEYS
                        }

                    before = env.read_physics()
                    _check_readback(before, physics)
                    if task in topology and topology[task] != before.topology_signature:
                        raise RuntimeError("Asset topology changed across physics values")
                    topology[task] = before.topology_signature

                    action = env.prepare_contact_probe()  # type: ignore[attr-defined]
                    initial_contact = env.contact_report()  # type: ignore[attr-defined]
                    maximum_contact_count = int(initial_contact["relevant_contact_count"])
                    maximum_contact_force = 0.0
                    maximum_counterpart_contacts = dict(
                        initial_contact["counterpart_contact_counts"]
                    )
                    for _ in range(steps):
                        observation, reward, terminated, truncated, info = env.step(action)
                        _check_observation(observation, env.image_size)
                        values = np.asarray(
                            [reward, *[float(value) for value in info.values()]], dtype=np.float64
                        )
                        if not np.isfinite(values).all():
                            raise RuntimeError("Non-finite reward or metric during smoke step")
                        contact = env.contact_report()  # type: ignore[attr-defined]
                        maximum_contact_count = max(
                            maximum_contact_count, int(contact["relevant_contact_count"])
                        )
                        maximum_contact_force = max(
                            maximum_contact_force, float(contact["maximum_contact_force"])
                        )
                        for name, count in contact["counterpart_contact_counts"].items():
                            maximum_counterpart_contacts[name] = max(
                                maximum_counterpart_contacts[name], int(count)
                            )
                        if terminated or truncated:
                            break
                    if maximum_contact_count < 1:
                        raise RuntimeError(f"No relevant {task} contact observed")
                    missing_counterparts = [
                        name for name, count in maximum_counterpart_contacts.items() if count < 1
                    ]
                    if missing_counterparts:
                        raise RuntimeError(
                            f"No {task} contact observed for counterparts {missing_counterparts}"
                        )
                    after = env.read_physics()
                    _check_readback(after, physics)

                    replay = env.reset()
                    if env.reset_fingerprint() != fingerprint:  # type: ignore[attr-defined]
                        raise RuntimeError("Same indexed reset did not replay its fingerprint")
                    replay_observation_deltas: dict[str, float] = {}
                    for obs_key in DEPLOYMENT_KEYS:
                        delta = _observation_delta(replay[obs_key], initial[obs_key])
                        replay_observation_deltas[obs_key] = delta
                        if delta > _OBSERVATION_TOLERANCES[obs_key]:
                            raise RuntimeError(f"Same indexed reset did not replay {obs_key}")
                    rows.append(
                        {
                            "task": task,
                            "physics_index": physics_index,
                            "episode_index": episode_index,
                            "steps_requested": steps,
                            "fingerprint": fingerprint,
                            "reset_state": state.as_dict(),
                            "initial_shapes": {
                                key: list(value.shape) for key, value in initial.items()
                            },
                            "readback_before": before.as_dict(),
                            "readback_after": after.as_dict(),
                            "maximum_relevant_contact_count": maximum_contact_count,
                            "maximum_relevant_contact_force": maximum_contact_force,
                            "maximum_counterpart_contact_counts": maximum_counterpart_contacts,
                            "paired_observation_max_abs_delta": paired_observation_deltas,
                            "replay_observation_max_abs_delta": replay_observation_deltas,
                            "config_hash": _config_hash(config),
                        }
                    )
            except Exception as error:  # noqa: BLE001 - preserve every smoke failure in JSON
                failures.append(f"{task}/physics-{physics_index}: {type(error).__name__}: {error}")
            finally:
                if env is not None:
                    env.close()

    for task in tasks:
        first = fingerprints.get((task, 0))
        second = fingerprints.get((task, 1))
        if first is not None and first == second:
            failures.append(f"{task}: reset indices 0 and 1 produced the same fingerprint")

    return {
        "status": "PASS" if not failures else "FAIL",
        "scope": "robosuite/MuJoCo plumbing smoke only; no training or performance claim",
        "renderer": os.environ.get("MUJOCO_GL", "unset"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": versions,
        "seed": seed,
        "observation_tolerances": _OBSERVATION_TOLERANCES,
        "tasks": tasks,
        "physics_values": [
            {"mass": physics.mass, "friction": physics.friction} for physics in physics_values
        ],
        "rows": rows,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-training robosuite/MuJoCo smoke gate")
    parser.add_argument("--task", choices=["all", "push", "lift"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/robosuite_smoke.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    tasks = ["push", "lift"] if args.task == "all" else [args.task]
    payload = smoke(tasks, seed=args.seed, steps=args.steps)
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
