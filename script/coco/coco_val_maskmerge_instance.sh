#!/usr/bin/env bash
set -euo pipefail

VARIANT="${VARIANT:-large}"
SIZE="${SIZE:-640}"
DEVICES="${DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-${1:-}}"
CKPT_PATH="${CKPT_PATH:-${2:-}}"
CONFIG="configs/maskmerge/dinov2/coco/instance/eomt_${VARIANT}_${SIZE}.yaml"

if [[ -z "${DATA_PATH}" || -z "${CKPT_PATH}" ]]; then
  echo "Usage: DATA_PATH=/path/to/coco CKPT_PATH=/path/to/pytorch_model.bin VARIANT=large SIZE=640 $0" >&2
  echo "Or: $0 /path/to/coco /path/to/pytorch_model.bin" >&2
  exit 2
fi

python3 main.py validate \
  -c "${CONFIG}" \
  --trainer.devices "${DEVICES}" \
  --data.batch_size "${BATCH_SIZE}" \
  --data.num_workers "${NUM_WORKERS}" \
  --data.path "${DATA_PATH}" \
  --model.ckpt_path "${CKPT_PATH}" \
  "${@:3}"
