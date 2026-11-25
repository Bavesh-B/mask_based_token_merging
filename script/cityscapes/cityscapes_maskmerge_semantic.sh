#!/usr/bin/env bash
set -euo pipefail

VARIANT="${VARIANT:-large}"
SIZE="${SIZE:-1024}"
DEVICES="${DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-${1:-}}"
CKPT_PATH="${CKPT_PATH:-${2:-}}"
CONFIG="configs/maskmerge/dinov2/cityscapes/semantic/eomt_${VARIANT}_${SIZE}.yaml"

if [[ -z "${DATA_PATH}" ]]; then
  echo "Usage: DATA_PATH=/path/to/cityscapes [CKPT_PATH=/path/to/pytorch_model.bin] VARIANT=large $0" >&2
  echo "Or: $0 /path/to/cityscapes [/path/to/pytorch_model.bin]" >&2
  exit 2
fi

cmd=(python3 main.py fit -c "${CONFIG}" --trainer.devices "${DEVICES}" --data.batch_size "${BATCH_SIZE}" --data.num_workers "${NUM_WORKERS}" --data.path "${DATA_PATH}")
if [[ -n "${CKPT_PATH}" ]]; then
  cmd+=(--model.ckpt_path "${CKPT_PATH}" --model.load_ckpt_class_head True)
fi
cmd+=("${@:3}")

"${cmd[@]}"
