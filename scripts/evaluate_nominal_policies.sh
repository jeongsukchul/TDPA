#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

for task in push lift; do
  python -m tdpa.evaluation.evaluate_nominal_policy \
    --mode competence --task "$task" \
    --checkpoint "artifacts/nominal/${task}_bc.pt" \
    --seeds 11 22 33 --episodes 20 \
    --output "artifacts/nominal/${task}_competence.json"
done

# Do not spend the OOD budget unless both task-specific competence gates pass.
for task in push lift; do
  command=(python -m tdpa.evaluation.evaluate_nominal_policy \
    --mode ood --task "$task" \
    --checkpoint "artifacts/nominal/${task}_bc.pt" \
    --competence-artifact "artifacts/nominal/${task}_competence.json" \
    --seeds 11 22 33 --episodes 20)
  output="artifacts/nominal/${task}_ood_gate.json"
  if [[ "$task" == "lift" ]]; then
    command+=(
      --lift-friction-validation artifacts/calibration/lift_friction_validation.json
    )
    output="artifacts/nominal/lift_ood_calibrated_v1.json"
  fi
  command+=(--output "$output")
  "${command[@]}"
done
