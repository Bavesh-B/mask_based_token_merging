python3 flops_fps.py \
  -c configs/token_merge/panoptic_finetune_large_640.yaml \
  --device cuda \
  --data.path /pub0/shared/joshua/ade20k \
  --ckpt checkpoints/stage1/panoptic_47.0.ckpt