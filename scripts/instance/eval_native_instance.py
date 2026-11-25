#!/usr/bin/env python3
"""Evaluate an instance checkpoint at an ARBITRARY resolution, decoupled from
the resolution it was trained/created at.

The trick: build the ViT encoder at its CREATION size (so the positional
embedding shape matches the checkpoint and weights load), but with
dynamic_img_size=True so timm interpolates the pos-embed per forward. Then set
the LightningModule's eval img_size to the desired eval resolution -- eval_step
resizes val images to that size, the encoder handles the larger token grid, and
mAP is computed at the eval resolution.

Lets us test the user's hypothesis: a model trained at 512x1024 can be
evaluated at native 1024x2048 (Mask2Former's Cityscapes protocol) for a big AP
gain on small/crowded objects, without retraining.

Run:
  python scripts/instance/eval_native_instance.py \
    --config configs/instance/cityscapes/eomt_base_512_finetune.yaml \
    --ckpt <run>/snapshots/epoch_final.ckpt \
    --create-size 512 1024 --eval-size 1024 2048 --tag dense-native
"""
from __future__ import annotations

import argparse
import copy
from importlib import import_module
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def instantiate_network(config):
    module, name = config["class_path"].rsplit(".", 1)
    args = {
        key: instantiate_network(value)
        if isinstance(value, dict) and "class_path" in value else value
        for key, value in config.get("init_args", {}).items()
    }
    return getattr(import_module(module), name)(**args)


def _tile_positions(dim, tile, stride):
    """Start coords covering [0, dim) with `tile`-sized windows at `stride`,
    clamping the final window so it ends exactly at `dim` (full coverage)."""
    if dim <= tile:
        return [0]
    pos = list(range(0, dim - tile + 1, stride))
    if pos[-1] != dim - tile:
        pos.append(dim - tile)
    return pos


@torch.no_grad()
def _mask_nms(masks, scores, labels, iou_thr):
    """Greedy per-class mask NMS. masks: bool [N,H,W] (native canvas).
    Returns kept indices. IoU computed against the (small) kept set only."""
    keep = []
    order = scores.argsort(descending=True)
    flat = masks.flatten(1).float()          # [N, H*W]
    areas = flat.sum(1)                       # [N]
    for idx in order.tolist():
        li = labels[idx]
        ok = True
        for k in keep:
            if labels[k] != li:
                continue
            inter = (flat[idx] * flat[k]).sum()
            union = areas[idx] + areas[k] - inter
            if union > 0 and (inter / union) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(idx)
    return keep


def _instances_from_logits(ml, cl, top_k, score_thr):
    """Turn per-query mask/class logits into (bmask[K,h,w], score[K], label[K]),
    replicating eval_step's topk + mask-score scoring, filtered by score_thr."""
    dev = ml.device
    scores = cl.softmax(dim=-1)[:, :-1]                # [Q, C]
    C = scores.shape[-1]
    labels = (torch.arange(C, device=dev)
              .unsqueeze(0).repeat(scores.shape[0], 1).flatten())
    topk_scores, topk_idx = scores.flatten(0, 1).topk(top_k, sorted=False)
    labels = labels[topk_idx]
    qidx = topk_idx // C
    ml_sel = ml[qidx]                                  # [top_k, h, w]
    bmask = ml_sel > 0
    mscore = (ml_sel.sigmoid().flatten(1) * bmask.flatten(1)).sum(1) / (
        bmask.flatten(1).sum(1) + 1e-6)
    inst = topk_scores * mscore
    keep = inst > score_thr
    return bmask[keep], inst[keep], labels[keep]


@torch.no_grad()
def _slide_predict(module, img, tile, overlap, top_k, nms_iou, score_thr,
                   max_keep, global_pass=False):
    """Sliding-window instance inference over one native uint8 image [3,H,W].
    Each tile is exactly the model's creation size (native pixel scale = the
    training scale), so there is no train/test scale mismatch. Per-tile
    instances are placed on the native canvas and merged with per-class mask
    NMS. When global_pass is set, also runs a whole-image pass (SAHI-style
    fusion) so large objects that span tiles are recovered cleanly. Returns a
    preds dict {masks,labels,scores} for the mAP metric."""
    _, H, W = img.shape
    tH, tW = tile
    sy = _tile_positions(H, tH, max(1, int(tH * (1 - overlap))))
    sx = _tile_positions(W, tW, max(1, int(tW * (1 - overlap))))

    cmasks, cscores, clabels = [], [], []

    if global_pass:
        # Whole image via the standard resize_and_pad path -> native-res masks.
        transformed = module.resize_and_pad_imgs_instance_panoptic([img])
        ml_layers, cl_layers = module(transformed)
        ml = F.interpolate(ml_layers[-1], module.img_size, mode="bilinear")
        ml = module.revert_resize_and_pad_logits_instance_panoptic(ml, [(H, W)])[0]
        gmask, ginst, glabels = _instances_from_logits(
            ml, cl_layers[-1][0], top_k, score_thr)
        for j in range(gmask.shape[0]):
            cmasks.append(gmask[j])
            cscores.append(ginst[j])
            clabels.append(glabels[j])

    for y0 in sy:
        for x0 in sx:
            crop = img[:, y0:y0 + tH, x0:x0 + tW]
            transformed = module.resize_and_pad_imgs_instance_panoptic([crop])
            ml_layers, cl_layers = module(transformed)
            ml = F.interpolate(ml_layers[-1], (tH, tW), mode="bilinear")[0]
            bmask, inst, labels = _instances_from_logits(
                ml, cl_layers[-1][0], top_k, score_thr)
            for j in range(bmask.shape[0]):
                canvas = torch.zeros((H, W), dtype=torch.bool, device=img.device)
                canvas[y0:y0 + tH, x0:x0 + tW] = bmask[j]
                cmasks.append(canvas)
                cscores.append(inst[j])
                clabels.append(labels[j])

    if not cmasks:
        return dict(masks=torch.zeros((0, H, W), dtype=torch.bool, device=img.device),
                    labels=torch.zeros((0,), dtype=torch.long, device=img.device),
                    scores=torch.zeros((0,), device=img.device))
    masks = torch.stack(cmasks)
    scores = torch.stack(cscores)
    labels = torch.stack(clabels)
    kept = _mask_nms(masks, scores, labels, nms_iou)
    kept = kept[:max_keep]
    idx = torch.tensor(kept, device=img.device)
    return dict(masks=masks[idx], labels=labels[idx], scores=scores[idx])


def build_module(config_path, ckpt, device, num_classes, create_size,
                 eval_size, overrides):
    from training.ideas_mask_classification_instance import (
        IdeasMaskClassificationInstance,
    )
    cfg = yaml.safe_load((ROOT / config_path).read_text())
    net_cfg = copy.deepcopy(cfg["model"]["init_args"]["network"])
    net_cfg.setdefault("init_args", {})
    net_cfg["init_args"].setdefault("num_classes", num_classes)
    net_cfg["init_args"].update(overrides)
    enc = net_cfg["init_args"]["encoder"]
    enc.setdefault("init_args", {})
    # Encoder built at the CREATION size (pos-embed shape matches the ckpt),
    # with dynamic_img_size so it can run at eval_size via pos-embed interp.
    enc["init_args"]["img_size"] = list(create_size)
    enc["init_args"]["dynamic_img_size"] = True
    enc["init_args"]["ckpt_path"] = str(ckpt)
    if "merge_conf_threshold_end" in net_cfg["init_args"] and net_cfg["init_args"].get(
        "merge_schedule_enabled", True
    ):
        net_cfg["init_args"]["merge_conf_threshold"] = net_cfg["init_args"].get(
            "merge_conf_threshold_end", 0.0
        )
    network = instantiate_network(net_cfg)
    module = IdeasMaskClassificationInstance(
        network=network, img_size=tuple(eval_size), num_classes=num_classes,
        attn_mask_annealing_enabled=False,
        ckpt_path=str(ckpt),
    )
    return module.to(device).eval()


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", choices=["cityscapes", "isaid"], default="cityscapes")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--create-size", type=int, nargs=2, default=[512, 1024])
    ap.add_argument("--eval-size", type=int, nargs=2, default=[1024, 2048])
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = full val set")
    ap.add_argument("--mode", choices=["whole", "slide"], default="whole")
    ap.add_argument("--tile", type=int, nargs=2, default=[512, 1024])
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--global-pass", action="store_true",
                    help="SAHI-style: also run whole-image pass, fuse via NMS")
    ap.add_argument("--algm-threshold", type=float, default=None)
    ap.add_argument("--algm-layers", default="")
    args = ap.parse_args()
    device = torch.device(args.device)

    overrides = {}
    if args.algm_layers:
        overrides["algm_selected_layers"] = [int(x) for x in args.algm_layers.split(",")]
    if args.algm_threshold is not None:
        overrides["local_merge_threshold"] = args.algm_threshold

    if args.dataset == "isaid":
        from datasets.isaid_instance import ISAIDInstance
        dm = ISAIDInstance(path=args.data_path, batch_size=args.batch_size,
                           num_workers=args.num_workers, img_size=tuple(args.eval_size),
                           num_classes=args.num_classes)
    else:
        from datasets.cityscapes_instance import CityscapesInstance
        dm = CityscapesInstance(path=args.data_path, batch_size=args.batch_size,
                                num_workers=args.num_workers, val_size=None)
    dm.setup("validate")
    module = build_module(args.config, args.ckpt, device, args.num_classes,
                          args.create_size, args.eval_size, overrides)

    loader = dm.val_dataloader()
    n = 0
    for batch in loader:
        imgs, targets = batch
        imgs = [img.to(device) for img in imgs]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            if args.mode == "slide":
                for img, tgt in zip(imgs, targets):
                    preds = _slide_predict(
                        module, img, tuple(args.tile), args.overlap,
                        args.top_k, args.nms_iou, args.score_thr,
                        module.eval_top_k_instances,
                        global_pass=args.global_pass)
                    module.metrics[-1].update([preds], [dict(
                        masks=tgt["masks"], labels=tgt["labels"],
                        iscrowd=tgt["is_crowd"])])
            else:
                module.eval_step((imgs, targets))
        n += len(imgs)
        if args.limit and n >= args.limit:
            break

    res = module.metrics[-1].compute()
    m = {k: (v.item() if torch.is_tensor(v) and v.ndim == 0 else v)
         for k, v in res.items()}
    tag_mode = f"{args.mode}" + (
        f" tile{tuple(args.tile)} ov{args.overlap}" if args.mode == "slide" else "")
    print(f"\n=== {args.tag} | {tag_mode} | create {tuple(args.create_size)} -> "
          f"eval {tuple(args.eval_size)} | {n} val imgs ===")
    print(f"  mAP        {m.get('map', float('nan')):.4f}")
    print(f"  mAP@50     {m.get('map_50', float('nan')):.4f}")
    print(f"  mAP@75     {m.get('map_75', float('nan')):.4f}")
    print(f"  mAP small  {m.get('map_small', float('nan')):.4f}")
    print(f"  mAP medium {m.get('map_medium', float('nan')):.4f}")
    print(f"  mAP large  {m.get('map_large', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
