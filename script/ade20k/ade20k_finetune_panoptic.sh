# wandb settings
export WANDB_API_KEY='bc3a85a50abf745031ae3e355d7b9b8c61bb359c'
# export WANDB_MODE=disabled

# DDP settings
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO

# Finetune
python3 main.py fit \
  -c configs/token_merge/panoptic_finetune_large_640.yaml \
  --trainer.devices 2 \
  --data.batch_size 8 \
  --data.path /pub0/shared/joshua/ade20k \
  --model.ckpt_path checkpoints/pretrained_eomt/ade20k_ps.bin \
  --model.load_ckpt_class_head False