#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SERVER_HOST:?Set SERVER_HOST to the learner address}"
source client/.venv/bin/activate
python -m client.run_client --host="$SERVER_HOST" --port=8102 \
  --config_task_path=configs/task/ur_insert.py "$@"
