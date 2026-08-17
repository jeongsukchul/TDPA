#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

checkpoint="${TDPA_LIFT_CHECKPOINT:-artifacts/nominal/lift_bc_spatial.pt}"
competence="${TDPA_LIFT_COMPETENCE:-artifacts/nominal/lift_competence_spatial.json}"

python -m tdpa.evaluation.evaluate_nominal_policy \
  --mode ood \
  --task lift \
  --checkpoint "$checkpoint" \
  --competence-artifact "$competence" \
  --lift-friction-validation artifacts/calibration/lift_friction_validation.json \
  --seeds 11 22 33 \
  --episodes 20 \
  --output artifacts/nominal/lift_ood_calibrated_v1.json
