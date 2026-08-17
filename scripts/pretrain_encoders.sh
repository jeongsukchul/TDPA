#!/usr/bin/env bash
set -euo pipefail

for variant in response distill hybrid; do
  python -m tdpa.training.train_encoder \
    --variant "$variant" \
    --datasets artifacts/interactions_push.npz artifacts/interactions_lift.npz \
    --epochs "${TDPA_EPOCHS:-10}" \
    --output "artifacts/encoder_${variant}.pt"
done

