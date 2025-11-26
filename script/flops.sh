python3 flops_fps.py \
  -c configs/token_merge/stage2/semantic_fromscratch_large_512.yaml \
  --device cuda:2 \
  --data.path /pub0/shared/joshua/ade20k \
  --ckpt checkpoints/last-v1.ckpt