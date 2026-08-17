#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -m tdpa.evaluation.lift_feasibility \
  --mode development \
  --output artifacts/nominal/lift_feasibility_development.json \
  --strict
