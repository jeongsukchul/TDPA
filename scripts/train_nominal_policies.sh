#!/usr/bin/env bash
set -euo pipefail

epochs="${TDPA_NOMINAL_EPOCHS:-100}"

python -m tdpa.training.train_nominal_policy \
  --task push --dataset artifacts/nominal/push_demos.hdf5 \
  --epochs "$epochs" --device auto --output artifacts/nominal/push_bc.pt
python -m tdpa.training.train_nominal_policy \
  --task lift --dataset artifacts/nominal/lift_demos.hdf5 \
  --epochs "$epochs" --device auto --output artifacts/nominal/lift_bc.pt
