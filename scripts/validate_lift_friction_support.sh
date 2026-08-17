#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -m tdpa.evaluation.lift_friction_validation \
  --mode validation \
  --output artifacts/calibration/lift_friction_validation.json \
  --strict
