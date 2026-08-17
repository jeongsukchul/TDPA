from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdpa.evaluation.evaluate_ood import evaluate
from tdpa.training.train_adapter import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task-specific adaptation-data curves")
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--base-episodes", type=int, default=100)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/adaptation_curve.json"))
    args = parser.parse_args()
    rows = []
    checkpoint_dir = args.output.parent / "curve_checkpoints"
    for seed in args.seeds:
        for fraction in args.fractions:
            if not 0 < fraction <= 1:
                raise ValueError("Every data fraction must be in (0,1]")
            episodes = max(1, round(args.base_episodes * fraction))
            checkpoint = checkpoint_dir / f"{args.task}_seed{seed}_fraction{fraction:g}.pt"
            training = train(
                task=args.task,
                encoder=args.encoder,
                output=checkpoint,
                episodes=episodes,
                epochs=args.epochs,
                batch_size=128,
                seed=seed,
                device=args.device,
            )
            evaluation = evaluate(
                args.task,
                args.encoder,
                checkpoint,
                [seed],
                args.eval_episodes,
                args.device,
            )
            for split, metrics in evaluation["summary"].items():
                rows.append(
                    {
                        "task": args.task,
                        "method": "pretrained_shared",
                        "seed": seed,
                        "fraction": fraction,
                        "task_specific_episodes": episodes,
                        "task_specific_samples": training["samples"],
                        "split": split,
                        **metrics,
                    }
                )
    payload = {
        "x_axis": "task_specific_episodes",
        "y_axis": "success_rate",
        "pretraining_cost_excluded_from_x_axis": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {len(rows)} curve points to {args.output}")


if __name__ == "__main__":
    main()

