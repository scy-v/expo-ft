#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source expo-setup/activate.sh
python -m client.ur.evaluate --config="${UR_CONFIG:-configs/ur/insert_sft.yaml}" --serve "$@"
