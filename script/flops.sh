python3 flops_fps.py \
  -c configs/token_merge/stage3/panoptic_finetune_large_640.yaml \
  --device cuda:2 \
  --data.path /pub0/shared/joshua/ade20k \
  --ckpt checkpoints/last.ckpt