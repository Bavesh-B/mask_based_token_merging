# wandb settings
export WANDB_API_KEY='bc3a85a50abf745031ae3e355d7b9b8c61bb359c'
# export WANDB_MODE=disabled

# Val
python3 main.py validate \
  -c configs/token_merge/panoptic_finetune_large_640.yaml \
  --trainer.devices 1 \
  --data.batch_size 16 \
  --data.path /pub0/shared/joshua/ade20k \
  --model.ckpt_path checkpoints/pretrained_eomt/ade20k_ps.bin