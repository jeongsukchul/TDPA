#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

push_checkpoint="${TDPA_PUSH_CHECKPOINT:-artifacts/nominal/push_bc.pt}"
push_competence="${TDPA_PUSH_COMPETENCE:-artifacts/nominal/push_competence.json}"
push_ood="${TDPA_PUSH_OOD:-artifacts/nominal/push_ood_gate.json}"
push_output="${TDPA_PUSH_ORACLE_OUTPUT:-artifacts/nominal/push_oracle_gate.json}"

lift_checkpoint="${TDPA_LIFT_CHECKPOINT:-artifacts/nominal/lift_bc_spatial.pt}"
lift_competence="${TDPA_LIFT_COMPETENCE:-artifacts/nominal/lift_competence_spatial.json}"
lift_ood="${TDPA_LIFT_OOD:-artifacts/nominal/lift_ood_spatial.json}"
lift_output="${TDPA_LIFT_ORACLE_OUTPUT:-artifacts/nominal/lift_oracle_gate.json}"

python -m tdpa.evaluation.robosuite_oracle_gate \
  --mode full --task push \
  --checkpoint "$push_checkpoint" \
  --competence-artifact "$push_competence" \
  --ood-artifact "$push_ood" \
  --output "$push_output"

python -m tdpa.evaluation.robosuite_oracle_gate \
  --mode full --task lift \
  --checkpoint "$lift_checkpoint" \
  --competence-artifact "$lift_competence" \
  --ood-artifact "$lift_ood" \
  --output "$lift_output"

python - "$push_output" "$lift_output" <<'PY'
import json
import sys
from pathlib import Path

failed = []
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    print(f"{artifact['task']}: {artifact['status']} -> {path}")
    if artifact["status"] != "PASS":
        failed.append(artifact["task"])
if failed:
    raise SystemExit(f"Oracle gate failed for: {', '.join(failed)}")
PY
