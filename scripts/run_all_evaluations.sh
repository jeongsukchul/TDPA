#!/usr/bin/env bash
set -euo pipefail

python -m tdpa.evaluation.oracle_gate --task all --seeds 0 1 2 --episodes 20 --strict
for task in push lift; do
  python -m tdpa.evaluation.evaluate_ood \
    --task "$task" \
    --encoder artifacts/encoder_hybrid_student.pt \
    --adapter "artifacts/adapter_${task}.pt" \
    --seeds 0 1 2 \
    --episodes 20 \
    --output "artifacts/ood_${task}.json"
done
python -m tdpa.evaluation.representation_probe \
  --encoder artifacts/encoder_hybrid.pt \
  --dataset \
    artifacts/interactions_push_id.npz \
    artifacts/interactions_push_policy_shift.npz \
    artifacts/interactions_lift_id.npz \
    artifacts/interactions_lift_policy_shift.npz \
  --require-heldout \
  --output artifacts/representation_probe.json
