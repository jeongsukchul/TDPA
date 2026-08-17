#!/usr/bin/env bash
set -euo pipefail

for task in push lift; do
  python -m tdpa.training.train_adapter \
    --task "$task" \
    --encoder artifacts/encoder_hybrid_student.pt \
    --episodes "${TDPA_ADAPTER_EPISODES:-50}" \
    --epochs "${TDPA_EPOCHS:-10}" \
    --output "artifacts/adapter_${task}.pt"
done
