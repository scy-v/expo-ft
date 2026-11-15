#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source client/.venv/bin/activate
python -m client.ur.evaluate --config="${UR_CONFIG:-configs/ur/insert.yaml}" "$@"
