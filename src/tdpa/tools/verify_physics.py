from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

from tdpa.envs.base import Physics
from tdpa.envs.make_env import make_env
from tdpa.envs.physics_config import load_physics_config
from tdpa.envs.physics_randomization import PhysicsRandomizer, PhysicsSplit
from tdpa.utils.config import load_yaml


def verify(
    task: str, seed: int = 0, count: int = 3, *, backend: str = "synthetic"
) -> list[dict[str, object]]:
    randomizer = PhysicsRandomizer(load_physics_config(), seed)
    rows: list[dict[str, object]] = []
    for split in PhysicsSplit:
        for sample in randomizer.sample(split, count):
            env = make_env(
                task,
                physics=Physics(sample.mass, sample.friction),
                seed=seed,
                backend=backend,
            )
            try:
                readback = env.read_physics()
                geoms = readback.object_geoms + readback.counterpart_geoms
                mass_matches = abs(sample.mass - readback.actual_mass) < 1e-9
                friction_matches = all(
                    abs(sample.friction - geom.friction[0]) < 1e-9 for geom in geoms
                )
                if not (mass_matches and friction_matches):
                    raise RuntimeError("Live physics readback does not match requested values")
                rows.append(
                    {
                        **sample.as_dict(),
                        "task": task,
                        "backend": backend,
                        "readback": readback.as_dict(),
                    }
                )
            finally:
                env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep and read back physical parameters")
    parser.add_argument("--backend", choices=["synthetic", "robosuite"], default="synthetic")
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = verify(args.task, args.seed, args.count, backend=args.backend)
    task_config = load_yaml(f"configs/env/{args.task}.yaml")
    config_payload = json.dumps(task_config, sort_keys=True, separators=(",", ":"))
    versions = {"numpy": metadata.version("numpy")}
    if args.backend == "robosuite":
        versions.update(
            {
                "robosuite": metadata.version("robosuite"),
                "mujoco": metadata.version("mujoco"),
            }
        )
    payload = {
        "backend": args.backend,
        "python": sys.version.split()[0],
        "renderer": os.environ.get("MUJOCO_GL", "unset"),
        "versions": versions,
        "config_hash": hashlib.sha256(config_payload.encode("utf-8")).hexdigest(),
        "warning": (
            "MuJoCo ranges are plumbing-only and have not been calibrated."
            if args.backend == "robosuite"
            else "Synthetic readback."
        ),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
