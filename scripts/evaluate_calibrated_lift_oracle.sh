#!/usr/bin/env bash
set -euo pipefail

stage="${1:-development}"
if [[ "$stage" != "development" && "$stage" != "final" ]]; then
  echo "usage: $0 [development|final]" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
device="${TDPA_DEVICE:-cpu}"
checkpoint="${TDPA_LIFT_CHECKPOINT:-artifacts/nominal/lift_bc_spatial.pt}"
competence="${TDPA_LIFT_COMPETENCE:-artifacts/nominal/lift_competence_spatial.json}"
validation="${TDPA_LIFT_FRICTION_VALIDATION:-artifacts/calibration/lift_friction_validation.json}"
revision="$(python -c 'from tdpa.utils.config import load_yaml; print(load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")["revision"])')"
output="artifacts/nominal/lift_oracle_v2_r${revision}_liftcalv1_${stage}.json"

command=(
  python -m tdpa.evaluation.robosuite_oracle_v2
  --mode "$stage"
  --task lift
  --checkpoint "$checkpoint"
  --competence-artifact "$competence"
  --lift-friction-validation "$validation"
  --device "$device"
  --output "$output"
  --strict
)
if [[ "$stage" == "final" ]]; then
  command+=(
    --development-artifact "artifacts/nominal/lift_oracle_v2_r${revision}_liftcalv1_development.json"
  )
fi
"${command[@]}"
