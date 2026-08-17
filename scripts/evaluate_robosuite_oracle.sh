#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/evaluate_robosuite_oracle_v2.sh" "$@"
