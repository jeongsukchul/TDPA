#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
episodes="${TDPA_NOMINAL_EPISODES:-200}"

python -m tdpa.data.collect_nominal_demos \
  --task push --episodes "$episodes" --seed 100 --index-start 0 \
  --output artifacts/nominal/push_demos.hdf5
python -m tdpa.data.collect_nominal_demos \
  --task lift --episodes "$episodes" --seed 200 --index-start 0 \
  --output artifacts/nominal/lift_demos.hdf5
