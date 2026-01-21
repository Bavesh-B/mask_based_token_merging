#!/usr/bin/env bash

# WANDB is optional; uncomment to disable network logging
export WANDB_API_KEY='bc3a85a50abf745031ae3e355d7b9b8c61bb359c'
# export WANDB_MODE=disabled

# CUDA visibility (set to a single GPU for a quick smoke test)
export CUDA_VISIBLE_DEVICES=2

# ADE20K data root; point to the folder that contains the downloaded zip/tar files
DATA_ROOT=/pub0/shared/joshua/ade20k

# DDP settings
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO

# Run a 1-step smoke test with ALGM config (panoptic fine-tune)
python3 main.py fit \
  -c configs/token_merge/stage3/semantic_finetune_large_512.yaml \
  --trainer.devices 1 \
  --data.batch_size 8 \
  --data.path "${DATA_ROOT}" \
  --model.ckpt_path checkpoints/pretrained_eomt/ade20k_ss.bin \
  --trainer.strategy ddp_find_unused_parameters_true
  # --model.load_ckpt_class_head False