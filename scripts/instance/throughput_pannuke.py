#!/usr/bin/env python3
"""Forward-only throughput over the FULL PanNuke fold-3 val set, in the setting
that matches the mAP measurement (batch 4, 640x640, fp16-mixed), so speed and
accuracy numbers are directly comparable.

Real images (not repeated) because MaskMerge and ALGM both merge
data-dependently -- the realised token count, and thus the speed, varies per
image. Times only the model forward (resize/pad + network + mask-logit
interpolate = the eval_step compute), excluding data loading and the mAP
metric, with CUDA sync and warmup.

Run:
  python scripts/instance/throughput_pannuke.py \
    --config configs/instance/pannuke/maskmerge_stage2.yaml \
    --ckpt <mm_stage2>/snapshots/epoch_final.ckpt --tag maskmerge
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.instance.eval_native_instance import build_module as build_instance_module  # noqa: E402


def build_module(config_path, ckpt, device, num_classes, overrides):
    return build_instance_module(
        config_path, ckpt, device, num_classes, (640, 640), (640, 640), overrides
    )


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--algm-layers", default="")   # e.g. "1,5"
    ap.add_argument("--algm-threshold", type=float, default=None)
    ap.add_argument("--merge-start-layer", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    overrides = {}
    if args.algm_layers:
        overrides["algm_selected_layers"] = [int(x) for x in args.algm_layers.split(",")]
    if args.algm_threshold is not None:
        overrides["local_merge_threshold"] = args.algm_threshold
    if args.merge_start_layer is not None:
        overrides["merge_start_layer"] = args.merge_start_layer

    from datasets.pannuke_instance import PanNukeInstance
    dm = PanNukeInstance(path=args.data_path, batch_size=args.batch_size,
                         num_workers=8)
    dm.setup("validate")
    module = build_module(args.config, args.ckpt, device, args.num_classes, overrides)

    loader = dm.val_dataloader()
    n_imgs, total_t = 0, 0.0
    step = 0
    for imgs, targets in loader:
        imgs = [img.to(device, non_blocking=True) for img in imgs]
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed = module.resize_and_pad_imgs_instance_panoptic(imgs)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            mask_logits_layers, class_logits_layers = module(transformed)
            ml = F.interpolate(mask_logits_layers[-1], module.img_size,
                               mode="bilinear")
            ml = module.revert_resize_and_pad_logits_instance_panoptic(ml, img_sizes)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step += 1
        if step > args.warmup:
            n_imgs += len(imgs)
            total_t += dt

    ips = n_imgs / total_t
    ms_per_img = 1000.0 * total_t / n_imgs
    print(f"{args.tag}: {ips:.2f} img/s  ({ms_per_img:.2f} ms/img)  "
          f"over {n_imgs} images, batch {args.batch_size}, fp16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
