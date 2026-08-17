#!/usr/bin/env bash
set -euo pipefail

python -m tdpa.training.train_base_policy --task push --output artifacts/base_push.json
python -m tdpa.training.train_base_policy --task lift --output artifacts/base_lift.json

