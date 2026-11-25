# MaskMerge on Mask2Former

This extension implements the approach in Section 4.4 on PanNuke with a DINOv2 ViT-B backbone. A frozen dense model supervises the coarse instance head, which supplies an instance map to the single-pass merge at inference.

## Setup

Use a separate environment with matching versions of PyTorch, torchvision, and CUDA. The experiment used Python 3.12, PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1, and Detectron2 0.6. Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), this directory's `requirements.txt`, and `pyarrow` for PanNuke conversion. From this directory, build the deformable-attention CUDA extension against the active PyTorch installation.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

## Evaluate released weights

Run from `extensions/mask2former`. Download the dense Mask2Former model and coarse head from the [model weights guide](../../MODEL_WEIGHTS.md#mask2former). The conversion command uses the three Parquet folds described in the [root README](../../README.md).

```bash
export PANNUKE_COCO_ROOT=/path/to/pannuke_coco
python convert_pannuke_coco.py /path/to/pannuke "$PANNUKE_COCO_ROOT"

CFG=configs/pannuke-instance/mask2former_dinov2b_pannuke.yaml
CKPT=/path/to/pannuke-mask2former-vit-b.pth
HEAD=/path/to/pannuke-mask2former-coarse-head-l9.pth
python train_pannuke.py --config-file "$CFG" --num-gpus 1 --eval-only MODEL.WEIGHTS "$CKPT"
python coarse_head_eval.py --layer 9 --gated --head "$HEAD" \
  --config-file "$CFG" --num-gpus 1 --eval-only MODEL.WEIGHTS "$CKPT"
```

Use `algm_full_probe.py` for local-then-global ALGM and `tome_probe.py` for fixed-count cosine merging. For the adaptive global cosine control, run `coarse_head_eval.py` with the same layer and threshold and omit `--gated`. Append `--bench` or `--flops` to the coarse-head evaluator before the trailing Detectron2 config overrides to time or count forward inference.

## Train the dense model and coarse head

```bash
python train_pannuke.py --config-file "$CFG" --num-gpus 1 OUTPUT_DIR output/pannuke
python coarse_head_train.py --layer 9 --iters 3000 --head-out output/pannuke/coarse_head.pth \
  --config-file "$CFG" MODEL.WEIGHTS output/pannuke/model_final.pth
```

The Mask2Former source is distributed under its [original license](LICENSE).
