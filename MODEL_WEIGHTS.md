# Model weights

Download the model you need from the [WACV 2027 GitHub release](https://github.com/Bavesh-B/mask_based_token_merging/releases/tag/wacv-2027). These files contain model weights for evaluation or initialization. Use the linked evaluation config with each MaskMerge download. Use the dense files with the matching dense EoMT or baseline configs.

## EoMT

| Model | MaskMerge weights | Dense weights | MaskMerge evaluation config |
|---|---|---|---|
| PanNuke / DINOv2 ViT-S | [MaskMerge](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-vit-s-maskmerge.pt) | [Dense](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-vit-s-dense.pt) | [YAML](configs/checkpoints/pannuke-vit-s.yaml) |
| PanNuke / DINOv2 ViT-B | [MaskMerge](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-vit-b-maskmerge.pt) | [Dense](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-vit-b-dense.pt) | [YAML](configs/checkpoints/pannuke-vit-b.yaml) |
| PanNuke / DINOv2 ViT-L | [MaskMerge](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-vit-l-maskmerge.pt) | [Dense](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-vit-l-dense.pt) | [YAML](configs/checkpoints/pannuke-vit-l.yaml) |
| PanNuke / MAE ViT-B | [MaskMerge](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-mae-vit-b-maskmerge.pt) | [Dense](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-mae-vit-b-dense.pt) | [YAML](configs/checkpoints/pannuke-mae-vit-b.yaml) |
| iSAID / DINOv2 ViT-B | [MaskMerge](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/isaid-vit-b-maskmerge.pt) | [Dense](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/isaid-vit-b-dense.pt) | [YAML](configs/checkpoints/isaid-vit-b.yaml) |
| Cityscapes / DINOv2 ViT-B | [MaskMerge](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/cityscapes-vit-b-maskmerge.pt) | [Dense](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/cityscapes-vit-b-dense.pt) | [YAML](configs/checkpoints/cityscapes-vit-b.yaml) |

Check downloads against [SHA256SUMS](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/SHA256SUMS).

## Evaluate a download

```bash
gh release download wacv-2027 --repo Bavesh-B/mask_based_token_merging \
  --pattern pannuke-vit-b-maskmerge.pt --dir checkpoints

python main.py validate -c configs/checkpoints/pannuke-vit-b.yaml \
  --data.path /path/to/pannuke --trainer.devices 1 \
  --model.ckpt_path checkpoints/pannuke-vit-b-maskmerge.pt
```

Use the same command with the iSAID config and weights for tiled validation. For Cityscapes, evaluate the downloaded model on the full validation images.

```bash
python -m scripts.instance.eval_native_instance --tag maskmerge \
  --config configs/checkpoints/cityscapes-vit-b.yaml \
  --ckpt checkpoints/cityscapes-vit-b-maskmerge.pt \
  --data-path /path/to/cityscapes --mode slide --tile 512 1024 --global-pass
```

## Mask2Former

The PanNuke extension uses [DINOv2 ViT-B Mask2Former weights](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-mask2former-vit-b.pth) and a [block-9 coarse instance head](https://github.com/Bavesh-B/mask_based_token_merging/releases/download/wacv-2027/pannuke-mask2former-coarse-head-l9.pth). The [extension guide](extensions/mask2former/README.md) includes the evaluation command.

For the training recipe described in the paper, see [Train](README.md#train).
