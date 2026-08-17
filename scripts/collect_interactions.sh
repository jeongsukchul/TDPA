#!/usr/bin/env bash
set -euo pipefail

python -m tdpa.data.interaction_collector --task push --episodes "${TDPA_EPISODES:-90}" --output artifacts/interactions_push.npz
python -m tdpa.data.interaction_collector --task lift --episodes "${TDPA_EPISODES:-90}" --output artifacts/interactions_lift.npz
for task in push lift; do
  python -m tdpa.data.interaction_collector --task "$task" --split id --episodes "${TDPA_DIAGNOSTIC_EPISODES:-18}" --output "artifacts/interactions_${task}_id.npz"
  python -m tdpa.data.interaction_collector --task "$task" --split policy_shift --episodes "${TDPA_DIAGNOSTIC_EPISODES:-18}" --output "artifacts/interactions_${task}_policy_shift.npz"
done
