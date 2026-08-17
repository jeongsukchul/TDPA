#!/usr/bin/env bash
set -euo pipefail

python -m tdpa.evaluation.oracle_gate --task all --seeds 0 1 2 --episodes 2 --strict
python -m tdpa.data.interaction_collector --task push --episodes 9 --episode-length 16 --output artifacts/smoke_push.npz
python -m tdpa.data.interaction_collector --task lift --episodes 9 --episode-length 16 --output artifacts/smoke_lift.npz
python -m tdpa.training.train_encoder --variant response --datasets artifacts/smoke_push.npz artifacts/smoke_lift.npz --epochs 1 --output artifacts/smoke_encoder.pt
python -m tdpa.training.train_adapter --task push --encoder artifacts/smoke_encoder_student.pt --episodes 2 --epochs 1 --output artifacts/smoke_push_adapter.pt
python -m tdpa.evaluation.evaluate_ood --task push --encoder artifacts/smoke_encoder_student.pt --adapter artifacts/smoke_push_adapter.pt --seeds 0 --episodes 1 --output artifacts/smoke_ood.json
