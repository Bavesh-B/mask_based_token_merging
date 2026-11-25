# Additional EoMT tasks

See the [release README](README.md) for the WACV instance segmentation experiments.

This tree includes a compact MaskMerge implementation in `models.maskmerge.EoMT`.
It follows the paper architecture:

- local 2x2 token merging after transformer layer 1
- temporary query insertion before layer 7
- object-level adaptive merging (OLAM) after layer 7
- mask confidence threshold annealing from 0.7 to 0 during fine-tuning
- query reinsertion before the final EoMT blocks
- sparse `base_to_current` index bookkeeping plus token sizes for weighted merge/unmerge

The selected local/global merge layers use ALGM-style placement when the backbone
block supports it: attention is run first, tokens are merged, then the block MLP is
run on the reduced token set. The implementation keeps compact index bookkeeping
instead of ALGM's dense source matrix. It also supports odd patch grids by merging
complete 2x2 windows and carrying remainder border tokens unchanged.

The default OLAM path uses `olam_mask_source: current`, which computes coarse mask
assignments directly on the currently merged tokens. This is the faster bookkeeping
path from the paper ablation. Set `olam_mask_source: full` to use the slower ablation
that unmerges to the full grid for the intermediate mask prediction.

The MaskMerge DINOv2 configs explicitly set `patch_size: 16`, matching the EoMT
checkpoints in this repo. Although the timm backbone names contain `patch14`,
EoMT adapts them with a 16x16 patch projection for the provided checkpoints.

## Configs

COCO panoptic:

```bash
configs/maskmerge/dinov2/coco/panoptic/eomt_small_640.yaml
configs/maskmerge/dinov2/coco/panoptic/eomt_base_640.yaml
configs/maskmerge/dinov2/coco/panoptic/eomt_large_640.yaml
configs/maskmerge/dinov2/coco/panoptic/eomt_large_1280.yaml
```

ADE20K panoptic:

```bash
configs/maskmerge/dinov2/ade20k/panoptic/eomt_small_640.yaml
configs/maskmerge/dinov2/ade20k/panoptic/eomt_base_640.yaml
configs/maskmerge/dinov2/ade20k/panoptic/eomt_large_640.yaml
configs/maskmerge/dinov2/ade20k/panoptic/eomt_large_1280.yaml
```

COCO instance:

```bash
configs/maskmerge/dinov2/coco/instance/eomt_small_640.yaml
configs/maskmerge/dinov2/coco/instance/eomt_base_640.yaml
configs/maskmerge/dinov2/coco/instance/eomt_large_640.yaml
configs/maskmerge/dinov2/coco/instance/eomt_large_1280.yaml
```

Cityscapes semantic:

```bash
configs/maskmerge/dinov2/cityscapes/semantic/eomt_small_1024.yaml
configs/maskmerge/dinov2/cityscapes/semantic/eomt_base_1024.yaml
configs/maskmerge/dinov2/cityscapes/semantic/eomt_large_1024.yaml
```

## Run

Fine-tune COCO panoptic:

```bash
DATA_PATH=/path/to/coco CKPT_PATH=/path/to/pytorch_model.bin VARIANT=large SIZE=640 \
  script/coco/coco_maskmerge_panoptic.sh
```

Fine-tune COCO instance:

```bash
DATA_PATH=/path/to/coco CKPT_PATH=/path/to/pytorch_model.bin VARIANT=large SIZE=640 \
  script/coco/coco_maskmerge_instance.sh
```

Fine-tune ADE20K panoptic:

```bash
DATA_PATH=/path/to/ade20k CKPT_PATH=/path/to/pytorch_model.bin VARIANT=large SIZE=640 \
  script/ade20k/ade20k_maskmerge_panoptic.sh
```

Fine-tune Cityscapes semantic:

```bash
DATA_PATH=/path/to/cityscapes CKPT_PATH=/path/to/pytorch_model.bin VARIANT=large \
  script/cityscapes/cityscapes_maskmerge_semantic.sh
```

Validation scripts use the same environment variables and require `CKPT_PATH`:

```bash
script/coco/coco_val_maskmerge_panoptic.sh /path/to/coco /path/to/pytorch_model.bin
script/ade20k/ade20k_val_maskmerge_panoptic.sh /path/to/ade20k /path/to/pytorch_model.bin
script/coco/coco_val_maskmerge_instance.sh /path/to/coco /path/to/pytorch_model.bin
script/cityscapes/cityscapes_val_maskmerge_semantic.sh /path/to/cityscapes /path/to/pytorch_model.bin
```
