#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SERVER_HOST="${SERVER_HOST:-100.101.84.9}"
source expo-setup/activate-client.sh
python -m client.run_client --host="$SERVER_HOST" --port=8102 \
  --config_task_path=configs/task/ur_insert.py "$@"
