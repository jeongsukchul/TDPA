#!/usr/bin/env bash
set -euo pipefail

stage="${1:-development}"
if [[ "$stage" != "development" && "$stage" != "final" ]]; then
  echo "usage: $0 [development|final]" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
device="${TDPA_DEVICE:-cpu}"
revision="$(python -c 'from tdpa.utils.config import load_yaml; print(load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml")["revision"])')"
revision_tag="r${revision}"

push_checkpoint="${TDPA_PUSH_CHECKPOINT:-artifacts/nominal/push_bc.pt}"
push_competence="${TDPA_PUSH_COMPETENCE:-artifacts/nominal/push_competence.json}"
lift_checkpoint="${TDPA_LIFT_CHECKPOINT:-artifacts/nominal/lift_bc_spatial.pt}"
lift_competence="${TDPA_LIFT_COMPETENCE:-artifacts/nominal/lift_competence_spatial.json}"

if [[ "$stage" == "final" ]]; then
  python - \
    "$push_checkpoint" "$push_competence" \
    "artifacts/nominal/push_oracle_v2_${revision_tag}_development.json" \
    "$lift_checkpoint" "$lift_competence" \
    "artifacts/nominal/lift_oracle_v2_${revision_tag}_development.json" <<'PY'
import sys
from pathlib import Path

from tdpa.evaluation.evaluate_nominal_policy import _json_hash, _validate_competence_artifact
from tdpa.evaluation.robosuite_oracle_v2 import _validate_development_artifact
from tdpa.policies.learned_nominal import checkpoint_sha256, current_environment_hash
from tdpa.utils.config import load_yaml

oracle_hash = _json_hash(load_yaml("configs/oracle/robosuite_perfect_context_v2.yaml"))
arguments = sys.argv[1:]
for task, offset in (("push", 0), ("lift", 3)):
    checkpoint = Path(arguments[offset])
    competence = Path(arguments[offset + 1])
    development = Path(arguments[offset + 2])
    checkpoint_hash = checkpoint_sha256(checkpoint)
    _validate_competence_artifact(competence, task=task, checkpoint_hash=checkpoint_hash)
    _validate_development_artifact(
        development,
        task=task,
        checkpoint_hash=checkpoint_hash,
        environment_hash=current_environment_hash(task),
        oracle_config_hash=oracle_hash,
        adapter_config_hash=_json_hash(load_yaml(f"configs/adapter/{task}.yaml")),
    )
print("Both development artifacts match the frozen oracle-v2 configuration.")
PY
fi

outputs=()
for task in push lift; do
  if [[ "$task" == "push" ]]; then
    checkpoint="$push_checkpoint"
    competence="$push_competence"
  else
    checkpoint="$lift_checkpoint"
    competence="$lift_competence"
  fi
  output="artifacts/nominal/${task}_oracle_v2_${revision_tag}_${stage}.json"
  command=(
    python -m tdpa.evaluation.robosuite_oracle_v2
    --mode "$stage"
    --task "$task"
    --checkpoint "$checkpoint"
    --competence-artifact "$competence"
    --device "$device"
    --output "$output"
  )
  if [[ "$stage" == "final" ]]; then
    command+=(
      --development-artifact "artifacts/nominal/${task}_oracle_v2_${revision_tag}_development.json"
    )
  fi
  "${command[@]}"
  outputs+=("$output")
done

python - "$stage" "${outputs[@]}" <<'PY'
import json
import sys
from pathlib import Path

stage = sys.argv[1]
failed = []
for raw_path in sys.argv[2:]:
    path = Path(raw_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    print(f"{stage} {artifact['task']}: {artifact['status']} -> {path}")
    if artifact["status"] != "PASS":
        failed.append(artifact["task"])
if failed:
    raise SystemExit(f"Oracle-v2 {stage} gate failed for: {', '.join(failed)}")
PY
