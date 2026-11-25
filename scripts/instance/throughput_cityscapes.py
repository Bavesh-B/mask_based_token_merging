#!/usr/bin/env python3
"""Forward-only throughput at 512x1024 (one SAHI tile) over real Cityscapes val
tiles, matching the accuracy setting's per-forward compute. Dense/ALGM/MaskMerge
merge data-dependently, so realised speed varies per tile -- we time real tiles,
batch 4, fp16, with warmup + CUDA sync, excluding data load and the mAP metric.

Run:
  python scripts/instance/throughput_cityscapes.py \
    --config configs/instance/cityscapes/maskmerge_512_long.yaml \
    --ckpt <mm>/last.ckpt --tag maskmerge
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.instance.eval_native_instance import build_module, _tile_positions  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--tile", type=int, nargs=2, default=[512, 1024])
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--max-tiles", type=int, default=400)
    ap.add_argument("--dataset", choices=["cityscapes", "isaid"], default="cityscapes")
    args = ap.parse_args()
    device = torch.device(args.device)
    tH, tW = args.tile
    need = args.max_tiles + args.batch_size * (args.warmup + 1)

    tiles = []
    if args.dataset == "isaid":
        from datasets.isaid_instance import ISAIDInstance
        dm = ISAIDInstance(path=args.data_path, batch_size=1, num_workers=6,
                           img_size=(tH, tW), num_classes=args.num_classes)
        dm.setup("validate")
        module = build_module(args.config, args.ckpt, device, args.num_classes,
                              [tH, tW], [tH, tW], {})
        # iSAID val items are ALREADY tiles at (tH, tW) -- take them directly.
        for imgs, _ in dm.val_dataloader():
            tiles.append(imgs[0])
            if len(tiles) >= need:
                break
    else:
        from datasets.cityscapes_instance import CityscapesInstance
        dm = CityscapesInstance(path=args.data_path, batch_size=1, num_workers=6,
                                val_size=None)
        dm.setup("validate")
        module = build_module(args.config, args.ckpt, device, args.num_classes,
                              [512, 1024], list(args.tile), {})
        # Collect real tiles from full val images (SAHI tiling).
        for imgs, _ in dm.val_dataloader():
            img = imgs[0]
            _, H, W = img.shape
            for y0 in _tile_positions(H, tH, max(1, int(tH * (1 - args.overlap)))):
                for x0 in _tile_positions(W, tW, max(1, int(tW * (1 - args.overlap)))):
                    tiles.append(img[:, y0:y0 + tH, x0:x0 + tW])
                    if len(tiles) >= need:
                        break
                if len(tiles) >= need:
                    break
            if len(tiles) >= need:
                break

    n_done, total_t = 0, 0.0
    step = 0
    for b in range(0, len(tiles) - args.batch_size, args.batch_size):
        batch = [t.to(device) for t in tiles[b:b + args.batch_size]]
        transformed = module.resize_and_pad_imgs_instance_panoptic(batch)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            module(transformed)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step += 1
        if step > args.warmup:
            n_done += len(batch)
            total_t += dt
        if n_done >= args.max_tiles:
            break

    ips = n_done / total_t
    print(f"{args.tag}: {ips:.2f} tiles/s  ({1000.0*total_t/n_done:.2f} ms/tile)  "
          f"over {n_done} tiles @ {tH}x{tW}, batch {args.batch_size}, fp16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
