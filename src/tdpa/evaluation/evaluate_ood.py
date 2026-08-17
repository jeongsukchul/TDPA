from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tdpa.evaluation.evaluate_task import load_adapter, run_adapted_episode
from tdpa.evaluation.metrics import aggregate_episode_metrics
from tdpa.evaluation.oracle_gate import make_manifest, manifest_hash
from tdpa.models.bundle import load_deployment_encoder
from tdpa.utils.checkpoints import file_sha256, git_commit, runtime_versions


def evaluate(
    task: str,
    encoder_path: Path,
    adapter_path: Path,
    seeds: list[int],
    episodes: int,
    device: str = "cpu",
) -> dict[str, object]:
    manifest = make_manifest(task, seeds, episodes)
    bundle = load_deployment_encoder(str(encoder_path), device)
    adapter = load_adapter(adapter_path, device)
    rows = [run_adapted_episode(row, bundle, adapter, device) for row in manifest]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)
    return {
        "task": task,
        "method": "pretrained_shared",
        "manifest_hash": manifest_hash(manifest),
        "summary": {split: aggregate_episode_metrics(values) for split, values in sorted(grouped.items())},
        "episodes": rows,
        "reproducibility": {
            "git_commit": git_commit(),
            "seeds": seeds,
            "physics_configuration": ["configs/physics/train.yaml", "configs/physics/ood.yaml"],
            "encoder_checkpoint": str(encoder_path),
            "encoder_hash": file_sha256(encoder_path),
            "adapter_checkpoint": str(adapter_path),
            "adapter_hash": file_sha256(adapter_path),
            "normalization_statistics": bundle.config.get("normalization", {}),
            "evaluation_splits": sorted(grouped),
            "versions": runtime_versions(),
        },
        "warning": "Synthetic infrastructure result only; not a robotics claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen encoder and task adapter")
    parser.add_argument("--task", choices=["push", "lift"], required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ood_evaluation.json"))
    args = parser.parse_args()
    result = evaluate(
        args.task, args.encoder, args.adapter, args.seeds, args.episodes, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
