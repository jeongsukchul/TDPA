#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

device="${TDPA_DEVICE:-cpu}"
checkpoint="${TDPA_LIFT_CHECKPOINT:-artifacts/nominal/lift_bc_spatial.pt}"
competence="${TDPA_LIFT_COMPETENCE:-artifacts/nominal/lift_competence_spatial.json}"
validation="${TDPA_LIFT_FRICTION_VALIDATION:-artifacts/calibration/lift_friction_validation.json}"
source="${TDPA_LIFT_ORACLE_R3_DEVELOPMENT:-artifacts/nominal/lift_oracle_v2_r3_liftcalv1_development.json}"
output="${TDPA_LIFT_SPATIAL_ORACLE_OUTPUT:-artifacts/nominal/lift_spatial_residual_oracle_v1_development.json}"

python -m tdpa.evaluation.robosuite_spatial_residual_gate \
  --mode development \
  --checkpoint "$checkpoint" \
  --competence-artifact "$competence" \
  --lift-friction-validation "$validation" \
  --source-oracle-development "$source" \
  --device "$device" \
  --output "$output" \
  --strict
