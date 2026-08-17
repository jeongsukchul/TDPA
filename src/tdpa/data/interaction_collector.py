from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tdpa.controllers.adapter_action_mapper import AdapterActionMapper
from tdpa.data.sequence_dataset import FixedFormatEpisode
from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import load_physics_config
from tdpa.envs.physics_randomization import PhysicsRandomizer, PhysicsSplit
from tdpa.utils.config import load_yaml

PROBE_PRIMITIVES = (
    "free_motion",
    "short_push",
    "short_pull",
    "hold",
    "lift_pulse",
    "vertical_load",
    "lateral_load",
    "surface_press",
    "short_slide",
)


def _probe_action(
    primitive: str, step: int, horizon: int, policy_id: str
) -> np.ndarray:
    phase = step / max(horizon - 1, 1)
    action = np.zeros(4, dtype=np.float32)
    # All contact probes start by approaching the visually identical object.
    approach_fraction = 0.28
    if primitive != "free_motion" and phase < approach_fraction:
        action[:] = (1.0, 0.0, -0.15, -1.0)
        return action
    wave = float(np.sin(2.0 * np.pi * 2.0 * phase))
    table = {
        "free_motion": (0.3 * wave, 0.25, 0.2 * wave, -1.0),
        "short_push": (0.8 if phase < 0.7 else 0.0, 0.0, 0.0, -1.0),
        "short_pull": (-0.6, 0.0, 0.0, -1.0),
        "hold": (0.0, 0.0, 0.0, 1.0),
        "lift_pulse": (0.0, 0.0, 0.8 if phase < 0.65 else 0.0, 1.0),
        "vertical_load": (0.0, 0.0, -0.45, 1.0),
        "lateral_load": (0.0, 0.55 * np.sign(wave or 1.0), 0.0, 1.0),
        "surface_press": (0.15, 0.0, -0.55, -1.0),
        "short_slide": (0.35, 0.5 * wave, -0.15, 1.0),
    }
    action[:] = table[primitive]
    if policy_id == "chirp":
        action[:3] += 0.12 * np.sin(2.0 * np.pi * (1.0 + 3.0 * phase) * phase)
    return np.clip(action, -1.0, 1.0)


def collect_interactions(
    task: str,
    episodes: int,
    *,
    episode_length: int = 48,
    seed: int = 0,
    split: PhysicsSplit | str = PhysicsSplit.TRAIN,
) -> list[FixedFormatEpisode]:
    if episodes <= 0 or episode_length < 3:
        raise ValueError("episodes must be positive and episode_length must be at least 3")
    split = PhysicsSplit(split)
    randomizer = PhysicsRandomizer(load_physics_config(), seed)
    mapper = AdapterActionMapper(load_yaml(f"configs/adapter/{task}.yaml"))
    output: list[FixedFormatEpisode] = []
    for episode_index in range(episodes):
        primitive = PROBE_PRIMITIVES[episode_index % len(PROBE_PRIMITIVES)]
        # Reuse each physics configuration across all primitives to defeat an
        # action-distribution shortcut.
        physics_index = episode_index // len(PROBE_PRIMITIVES)
        sample = randomizer.sample_at(split, physics_index)
        env = make_env(task, physics=Physics(sample.mass, sample.friction), seed=seed + episode_index)
        observation = env.reset()
        rgbd = np.zeros((episode_length, 4, env.image_size, env.image_size), dtype=np.float32)
        proprio = np.zeros((episode_length, 10), dtype=np.float32)
        actions = np.zeros((episode_length, 7), dtype=np.float32)
        responses = np.zeros((episode_length, 12), dtype=np.float32)
        privileged = np.zeros((episode_length, 16), dtype=np.float32)
        for step in range(episode_length):
            rgbd[step] = observation["rgbd"]
            proprio[step] = observation["proprio"]
            privileged[step] = env.privileged_observation()
            action = _probe_action(primitive, step, episode_length, sample.behavior_policy_id)
            applied = mapper.apply(action)
            if step + 1 < episode_length:
                actions[step] = applied.execution_command
                observation, _, _, _, _ = env.step(applied.action, applied.controller)
                responses[step + 1] = env.response()
        output.append(
            FixedFormatEpisode(
                rgbd=rgbd,
                proprio=proprio,
                actions=actions,
                responses=responses,
                privileged={"teacher_history": privileged},
                metadata={
                    "task": task,
                    "probe_id": primitive,
                    "policy_id": sample.behavior_policy_id,
                    "physics": {"mass": sample.mass, "friction": sample.friction},
                    "split": split.value,
                    "seed": seed,
                    "episode_index": episode_index,
                    "physics_sample_index": physics_index,
                },
            )
        )
    return output


def save_episode_archive(path: str | Path, episodes: list[FixedFormatEpisode]) -> None:
    if not episodes:
        raise ValueError("Cannot save an empty archive")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        rgbd=np.stack([episode.rgbd.numpy() for episode in episodes]),
        proprio=np.stack([episode.proprio.numpy() for episode in episodes]),
        actions=np.stack([episode.actions.numpy() for episode in episodes]),
        responses=np.stack([episode.responses.numpy() for episode in episodes]),
        privileged=np.stack(
            [episode.privileged["teacher_history"].numpy() for episode in episodes]
        ),
        metadata_json=np.asarray(
            json.dumps([episode.metadata for episode in episodes], sort_keys=True)
        ),
        format_version=np.asarray("tdpa-interaction-v1"),
    )


def load_episode_archive(path: str | Path) -> list[FixedFormatEpisode]:
    with np.load(Path(path), allow_pickle=False) as data:
        version = str(data["format_version"].item())
        if version != "tdpa-interaction-v1":
            raise ValueError(f"Unsupported dataset format: {version}")
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: data[key].copy() for key in ("rgbd", "proprio", "actions", "responses", "privileged")}
    count = arrays["actions"].shape[0]
    if len(metadata) != count:
        raise ValueError("Metadata/episode count mismatch")
    return [
        FixedFormatEpisode(
            rgbd=arrays["rgbd"][index],
            proprio=arrays["proprio"][index],
            actions=arrays["actions"][index],
            responses=arrays["responses"][index],
            privileged={"teacher_history": arrays["privileged"][index]},
            metadata=metadata[index],
        )
        for index in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect task-free physical interaction data")
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--episode-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=[item.value for item in PhysicsSplit], default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = collect_interactions(
        args.task,
        args.episodes,
        episode_length=args.episode_length,
        seed=args.seed,
        split=args.split,
    )
    save_episode_archive(args.output, episodes)
    print(f"saved {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
