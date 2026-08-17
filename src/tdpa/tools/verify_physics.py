from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import load_physics_config
from tdpa.envs.physics_randomization import PhysicsRandomizer, PhysicsSplit


def verify(task: str, seed: int = 0, count: int = 3) -> list[dict[str, object]]:
    randomizer = PhysicsRandomizer(load_physics_config(), seed)
    rows: list[dict[str, object]] = []
    for split in PhysicsSplit:
        for sample in randomizer.sample(split, count):
            env = make_env(task, physics=Physics(sample.mass, sample.friction), seed=seed)
            actual = env.physics
            if not (
                abs(sample.mass - actual.mass) < 1e-9
                and abs(sample.friction - actual.friction) < 1e-9
            ):
                raise RuntimeError("Physics readback does not match the requested values")
            rows.append(
                {
                    **sample.as_dict(),
                    "task": task,
                    "backend": "synthetic",
                    "requested_mass": sample.mass,
                    "actual_mass": actual.mass,
                    "requested_friction": sample.friction,
                    "actual_friction": actual.friction,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep and read back physical parameters")
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = verify(args.task, args.seed, args.count)
    payload = {
        "warning": "Synthetic readback only; MuJoCo task assets are not bundled.",
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

