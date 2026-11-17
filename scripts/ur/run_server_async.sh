#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source expo-setup/activate.sh
SFT_CKPT="${SFT_CKPT:-./checkpoints/sft/ur_insert/2500}"
DATASET_PATH="${DATASET_PATH:-./data/online_rl/success}"
ASSET_ID="${ASSET_ID:-scylearning/insert_black_rack_into_blue_slot_20260916_v01}"
# First visible GPU samples; remaining GPUs train: physical 2 samples, 0/1 train.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,0,1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Two training GPUs use batch32 each; allow more than JAX's default 75% memory cap.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export WANDB_MODE="${WANDB_MODE:-offline}"
python train_pi_robo_async.py \
  --config_task=configs/task/ur_insert.py \
  --config=configs/model/expo_ft_ur_config.py \
  --config.pi05_weight_loader_path="$SFT_CKPT/params" \
  --config.pi05_assets_dir="$SFT_CKPT/assets" \
  --config.pi05_asset_id="$ASSET_ID" \
  --dataset_path="$DATASET_PATH" --num_data=10 \
  --seed=42 --batch_size=64 --utd_ratio=20 --max_steps=100000 \
  --offline_ratio=0 --ep_timeout_secs=120 \
  --replan_steps=8 --delay=0 --fsdp_devices=1 \
  --client_host=0.0.0.0 --client_port=8102 \
  --project_name=expo_ft_ur --output_dir=./checkpoints/online_rl \
  --run_name="${RUN_NAME:-async_run_001}" --checkpoint_model --checkpoint_buffer \
  --checkpoint_interval=4000 "$@"
