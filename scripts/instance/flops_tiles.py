"""GFLOPs/tile for Cityscapes or iSAID, matching the accuracy/throughput setting.
Reuses the throughput script's tile loader + build_module, measures FLOPs with
torch's FlopCounterMode (one tile per forward, mean over --images tiles)."""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instance.eval_native_instance import build_module, _tile_positions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--dataset", choices=["cityscapes", "isaid"], required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--tile", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    tH, tW = args.tile

    tiles = []
    if args.dataset == "isaid":
        from datasets.isaid_instance import ISAIDInstance
        dm = ISAIDInstance(path=args.data_path, batch_size=1, num_workers=6,
                           img_size=(tH, tW), num_classes=args.num_classes)
        dm.setup("validate")
        module = build_module(args.config, args.ckpt, device, args.num_classes,
                              [tH, tW], [tH, tW], {})
        for imgs, _ in dm.val_dataloader():
            tiles.append(imgs[0])
            if len(tiles) >= args.images:
                break
    else:
        from datasets.cityscapes_instance import CityscapesInstance
        dm = CityscapesInstance(path=args.data_path, batch_size=1, num_workers=6,
                                val_size=None)
        dm.setup("validate")
        module = build_module(args.config, args.ckpt, device, args.num_classes,
                              [512, 1024], list(args.tile), {})
        for imgs, _ in dm.val_dataloader():
            img = imgs[0]
            _, H, W = img.shape
            for y0 in _tile_positions(H, tH, max(1, int(tH * (1 - args.overlap)))):
                for x0 in _tile_positions(W, tW, max(1, int(tW * (1 - args.overlap)))):
                    tiles.append(img[:, y0:y0 + tH, x0:x0 + tW])
                    if len(tiles) >= args.images:
                        break
                if len(tiles) >= args.images:
                    break
            if len(tiles) >= args.images:
                break

    total, n = 0.0, 0
    for t in tiles[:args.images]:
        transformed = module.resize_and_pad_imgs_instance_panoptic([t.to(device)])
        with FlopCounterMode(display=False) as fc:
            module(transformed)
        total += fc.get_total_flops() / 1e9
        n += 1
    print(f"{args.tag}: {total / n:.1f} GFLOPs/tile (mean over {n} tiles @ {tH}x{tW})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
