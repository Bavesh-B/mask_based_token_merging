# Reproducing the experiments

## MaskMerge training and evaluation

The [training configs](README.md#train) implement the paper's 15-epoch warm-start, with local merging at block 1 and OLAM at block 5 for ViT-S/B or block 8 for ViT-L. Block numbers are one-based. Use `--model.network.merge_conf_threshold 0.0` to validate the weights from your own completed run.

The [download guide](MODEL_WEIGHTS.md) lists the evaluation settings to use with each released weight file.

## Dense, ALGM, and ToMe

Validate the matching dense EoMT checkpoint using the corresponding dense config and `--model.network.masked_attn_enabled False`. For ALGM and ToMe, keep the same dense checkpoint and swap the config. Their configs use the reduced sequence in the later transformer blocks before reconstructing the spatial grid for mask readout.

| Dataset / backbone | ALGM | ToMe |
|---|---|---|
| PanNuke / ViT-B | [algm_mergedtail.yaml](configs/instance/pannuke/algm_mergedtail.yaml) | [mtome_r20.yaml](configs/instance/pannuke/mtome_r20.yaml) |
| PanNuke / ViT-S | [algm_mt_s.yaml](configs/instance/pannuke/algm_mt_s.yaml) | [tome_mt_s.yaml](configs/instance/pannuke/tome_mt_s.yaml) |
| PanNuke / ViT-L | [algm_mt_l.yaml](configs/instance/pannuke/algm_mt_l.yaml) | [tome_mt_l.yaml](configs/instance/pannuke/tome_mt_l.yaml) |
| PanNuke / MAE | [algm_mae.yaml](configs/instance/pannuke/mae/algm_mae.yaml) | [tome_mae.yaml](configs/instance/pannuke/mae/tome_mae.yaml) |
| iSAID / ViT-B | [algm_mt_isaid.yaml](configs/instance/isaid/algm_mt_isaid.yaml) | [mtome_isaid_L15_r30.yaml](configs/instance/isaid/mtome_isaid_L15_r30.yaml) |
| Cityscapes / ViT-B | [algm_mt_cityscapes.yaml](configs/instance/cityscapes/algm_mt_cityscapes.yaml) | [mtome_cs_L15_r30.yaml](configs/instance/cityscapes/mtome_cs_L15_r30.yaml) |

## Throughput and FLOPs

Run each variant on the same GPU, with the same batch size and input resolution. Forward throughput uses real validation inputs, discards warm-up batches, and synchronizes CUDA around timed forwards. Dataset loading and AP computation are outside the timed region. FLOPs are operator counts and their coverage can vary by PyTorch version.

```bash
MASKMERGE=configs/checkpoints/pannuke-vit-b.yaml
CKPT=checkpoints/pannuke-vit-b-maskmerge.pt
DATA=/path/to/pannuke

python -m scripts.instance.throughput_pannuke --config "$MASKMERGE" \
  --ckpt "$CKPT" --data-path "$DATA" --tag maskmerge --batch-size 4

python -m scripts.instance.flops_pannuke --config "$MASKMERGE" \
  --ckpt "$CKPT" --data-path "$DATA" --tag maskmerge
```

For Cityscapes use `scripts.instance.throughput_cityscapes` and `scripts.instance.flops_tiles` with `--dataset cityscapes --num-classes 8 --tile 512 1024`. For iSAID use the same scripts with `--dataset isaid --num-classes 15 --tile 512 512`.
