#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -m tdpa.evaluation.lift_friction_calibration \
  --stage refinement \
  --mode development \
  --output artifacts/calibration/lift_friction_refinement_development.json \
  --strict
