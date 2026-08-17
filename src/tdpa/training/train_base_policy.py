from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdpa.evaluation.oracle_gate import make_manifest, run_episode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and register the frozen synthetic nominal policy"
    )
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = next(item for item in make_manifest(args.task, [0], 1) if item.split == "nominal")
    result = run_episode(row, "no_adaptation")
    if not result["success"]:
        raise SystemExit("Nominal policy competence check failed")
    payload = {
        "type": "synthetic_visual_servo",
        "version": 1,
        "task": args.task,
        "frozen": True,
        "deployment_inputs": ["rgbd", "proprio"],
        "competence_check": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"registered frozen {args.task} policy at {args.output}")


if __name__ == "__main__":
    main()

