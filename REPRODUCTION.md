# Reproducing the experiments

## Checkpoints and layer placement

The primary MaskMerge configs use local merging at block 1 and OLAM at block 5 for ViT-S/B or block 8 for ViT-L. Block numbers are one-based. MaskMerge anneals its mask confidence threshold from 0.7 to 0 over the 15-epoch warm-start. Use `--model.network.merge_conf_threshold 0.0` for final Lightning validation.

The research workspace also contained the historical configs below. Use these when loading a checkpoint from those runs because layer placement and the number of query blocks affect predictions and checkpoint loading. Some of these runs used longer fine-tuning schedules than the camera-ready recipe.

| Dataset | Historical MaskMerge config |
|---|---|
| PanNuke ViT-B | [maskmerge_stage2.yaml](configs/instance/pannuke/maskmerge_stage2.yaml) |
| PanNuke ViT-S | [maskmerge_s.yaml](configs/instance/pannuke/maskmerge_s.yaml) |
| PanNuke ViT-L | [maskmerge_l_stage1.yaml](configs/instance/pannuke/maskmerge_l_stage1.yaml) |
| PanNuke MAE | [maskmerge_mae_warmstart.yaml](configs/instance/pannuke/mae/maskmerge_mae_warmstart.yaml) |
| iSAID | [maskmerge_isaid_warmstart.yaml](configs/instance/isaid/maskmerge_isaid_warmstart.yaml) |
| Cityscapes | [maskmerge_512_warmstart.yaml](configs/instance/cityscapes/maskmerge_512_warmstart.yaml) |

The reported values in the README are paper results. The release training commands and historical config paths document the workflow, and exact AP and throughput from a new run depend on the checkpoint, operating point, and hardware.

## Dense, ALGM, and ToMe

Validate the matching dense EoMT checkpoint using the corresponding dense config and `--model.network.masked_attn_enabled False`. For ALGM and ToMe, keep the same dense checkpoint and swap the config. Their configs use the reduced sequence in the later transformer blocks before reconstructing the spatial grid for mask readout.

| Dataset / backbone | ALGM | ToMe |
|---|---|---|
| PanNuke / ViT-B | [algm_mergedtail.yaml](configs/instance/pannuke/algm_mergedtail.yaml) | [mtome_r40.yaml](configs/instance/pannuke/mtome_r40.yaml) |
| PanNuke / ViT-S | [algm_mt_s.yaml](configs/instance/pannuke/algm_mt_s.yaml) | [tome_mt_s.yaml](configs/instance/pannuke/tome_mt_s.yaml) |
| PanNuke / ViT-L | [algm_mt_l.yaml](configs/instance/pannuke/algm_mt_l.yaml) | [tome_mt_l.yaml](configs/instance/pannuke/tome_mt_l.yaml) |
| PanNuke / MAE | [algm_mae.yaml](configs/instance/pannuke/mae/algm_mae.yaml) | [tome_mae.yaml](configs/instance/pannuke/mae/tome_mae.yaml) |
| iSAID / ViT-B | [algm_mt_isaid.yaml](configs/instance/isaid/algm_mt_isaid.yaml) | [mtome_isaid_L15_r30.yaml](configs/instance/isaid/mtome_isaid_L15_r30.yaml) |
| Cityscapes / ViT-B | [algm_mt_cityscapes.yaml](configs/instance/cityscapes/algm_mt_cityscapes.yaml) | [mtome_cs_L15_r30.yaml](configs/instance/cityscapes/mtome_cs_L15_r30.yaml) |

## Throughput and FLOPs

Run each variant on the same GPU, with the same batch size and input resolution. Forward throughput uses real validation inputs, discards warm-up batches, and synchronizes CUDA around timed forwards. Dataset loading and AP computation are outside the timed region. FLOPs are operator counts and their coverage can vary by PyTorch version.

```bash
python -m scripts.instance.throughput_pannuke --config "$MASKMERGE" \
  --ckpt "$CKPT" --data-path "$DATA" --tag maskmerge --batch-size 4

python -m scripts.instance.flops_pannuke --config "$MASKMERGE" \
  --ckpt "$CKPT" --data-path "$DATA" --tag maskmerge
```

For Cityscapes use `scripts.instance.throughput_cityscapes` and `scripts.instance.flops_tiles` with `--dataset cityscapes --num-classes 8 --tile 512 1024`. For iSAID use the same scripts with `--dataset isaid --num-classes 15 --tile 512 512`.
