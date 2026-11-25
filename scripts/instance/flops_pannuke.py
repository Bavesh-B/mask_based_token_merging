#!/usr/bin/env python3
"""Per-image forward GFLOPs on real PanNuke val images (data-dependent merging),
averaged. Same build/config path as throughput_pannuke.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.instance.throughput_pannuke import build_module  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--algm-threshold", type=float, default=None)
    ap.add_argument("--merge-start-layer", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    overrides = {}
    if args.algm_threshold is not None:
        overrides["local_merge_threshold"] = args.algm_threshold
    if args.merge_start_layer is not None:
        overrides["merge_start_layer"] = args.merge_start_layer

    from datasets.pannuke_instance import PanNukeInstance
    dm = PanNukeInstance(path=args.data_path, batch_size=1, num_workers=4)
    dm.setup("validate")
    module = build_module(args.config, args.ckpt, device, args.num_classes, overrides)

    total, n = 0.0, 0
    for imgs, targets in dm.val_dataloader():
        img = imgs[0].to(device)
        transformed = module.resize_and_pad_imgs_instance_panoptic([img])
        with FlopCounterMode(display=False) as fc:
            module(transformed)
        total += fc.get_total_flops() / 1e9
        n += 1
        if n >= args.images:
            break
    print(f"{args.tag}: {total / n:.1f} GFLOPs/image (mean over {n} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
