#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_from_class_path(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _instantiate(node: Any):
    if isinstance(node, dict) and "class_path" in node:
        cls = _import_from_class_path(node["class_path"])
        init_args = node.get("init_args", {})
        kwargs = {k: _instantiate(v) for k, v in init_args.items()}
        return cls(**kwargs)
    if isinstance(node, dict):
        return {k: _instantiate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def _to_device_batch(batch, device: torch.device, target_hw: tuple[int, int] | None = None):
    if len(batch) == 3:
        imgs, targets, entropy_maps = batch
    else:
        imgs, targets = batch
        entropy_maps = None

    # train_collate returns a tensor; eval_collate returns tuple/list of per-sample tensors.
    if isinstance(imgs, (list, tuple)):
        img_list = [im.to(device, non_blocking=True) for im in imgs]
        can_stack = all(im.shape == img_list[0].shape for im in img_list)
        imgs = torch.stack(img_list, dim=0) if can_stack else img_list
    else:
        imgs = imgs.to(device, non_blocking=True)

    # Ensure fixed spatial size expected by fixed-grid ViT backbones.
    if target_hw is not None:
        th, tw = int(target_hw[0]), int(target_hw[1])
        if isinstance(imgs, torch.Tensor):
            if imgs.shape[-2:] != (th, tw):
                imgs = F.interpolate(imgs, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            resized = []
            for im in imgs:
                if im.shape[-2:] != (th, tw):
                    im = F.interpolate(im.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)
                resized.append(im)
            imgs = resized

    if entropy_maps is not None:
        if isinstance(entropy_maps, dict):
            entropy_maps = {k: v.to(device, non_blocking=True) for k, v in entropy_maps.items()}
        elif isinstance(entropy_maps, (list, tuple)) and len(entropy_maps) > 0 and isinstance(entropy_maps[0], dict):
            em_list = [{k: v.to(device, non_blocking=True) for k, v in em.items()} for em in entropy_maps]
            patch_sizes = sorted(em_list[0].keys())
            can_stack = True
            for ps in patch_sizes:
                shape0 = em_list[0][ps].shape
                if any(em[ps].shape != shape0 for em in em_list):
                    can_stack = False
                    break
            if can_stack:
                entropy_maps = {ps: torch.stack([em[ps] for em in em_list], dim=0) for ps in patch_sizes}
            else:
                entropy_maps = em_list
        else:
            entropy_maps = None

    return imgs, targets, entropy_maps


def _normalize_config(cfg: dict, args):
    network_cfg = copy.deepcopy(cfg["model"]["init_args"]["network"])
    data_cfg = copy.deepcopy(cfg["data"])
    data_init = data_cfg.setdefault("init_args", {})

    if args.data_path is not None:
        data_init["path"] = args.data_path
    if "path" not in data_init:
        raise ValueError("Missing data path. Set data.init_args.path in config or pass --data-path.")

    if args.img_size is not None:
        img_size = tuple(args.img_size)
    else:
        img_size = tuple(data_init.get("img_size", (640, 640)))
    data_init["img_size"] = img_size

    net_init = network_cfg.setdefault("init_args", {})
    if "num_classes" not in net_init:
        if "num_classes" in data_init:
            net_init["num_classes"] = data_init["num_classes"]
        else:
            data_cls = _import_from_class_path(data_cfg["class_path"])
            sig = inspect.signature(data_cls.__init__)
            param = sig.parameters.get("num_classes")
            if param is not None and param.default is not inspect._empty:
                net_init["num_classes"] = param.default
            else:
                raise ValueError(
                    "Missing network num_classes. Set model.init_args.network.init_args.num_classes "
                    "or data.init_args.num_classes in config."
                )

    encoder_cfg = net_init.get("encoder")
    if isinstance(encoder_cfg, dict) and "class_path" in encoder_cfg:
        enc_init = encoder_cfg.setdefault("init_args", {})
        enc_init.setdefault("img_size", img_size)

    return network_cfg, data_cfg


def main():
    parser = argparse.ArgumentParser(description="Measure validation FPS for current EoMT config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/token_merge/stage3/panoptic_finetune_large_640.yaml"),
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    network_cfg, data_cfg = _normalize_config(cfg, args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    network = _instantiate(network_cfg).to(device).eval()
    dm = _instantiate(data_cfg)
    if args.num_workers is not None:
        dm.dataloader_kwargs["num_workers"] = args.num_workers
        dm.dataloader_kwargs["persistent_workers"] = args.num_workers > 0
    if args.batch_size is not None:
        dm.dataloader_kwargs["batch_size"] = args.batch_size
    dm.setup("fit")

    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    if device.type == "cuda" and args.amp != "off":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext

    total_model_s = 0.0
    total_e2e_s = 0.0
    total_images = 0

    total_iters = args.warmup + args.iters
    iterator = iter(loader)
    print(
        f"Running split={args.split} warmup={args.warmup} measured={args.iters} "
        f"batch_size={dm.dataloader_kwargs['batch_size']} amp={args.amp} device={device}"
    )

    for step in range(total_iters):
        t_e2e_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        imgs, _, entropy_maps = _to_device_batch(batch, device, target_hw=args.img_size)

        if device.type == "cuda":
            if isinstance(imgs, torch.Tensor):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    with amp_ctx():
                        _ = network(imgs, entropy_maps)
                end.record()
                torch.cuda.synchronize()
                model_s = start.elapsed_time(end) / 1000.0
                bsz = int(imgs.shape[0])
            else:
                model_s = 0.0
                bsz = len(imgs)
                for j, im in enumerate(imgs):
                    em = None
                    if isinstance(entropy_maps, list):
                        em = {k: v.unsqueeze(0) for k, v in entropy_maps[j].items()}
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.no_grad():
                        with amp_ctx():
                            _ = network(im.unsqueeze(0), em)
                    end.record()
                    torch.cuda.synchronize()
                    model_s += start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            if isinstance(imgs, torch.Tensor):
                with torch.no_grad():
                    _ = network(imgs, entropy_maps)
                model_s = time.perf_counter() - t0
                bsz = int(imgs.shape[0])
            else:
                model_s = 0.0
                bsz = len(imgs)
                for j, im in enumerate(imgs):
                    em = None
                    if isinstance(entropy_maps, list):
                        em = {k: v.unsqueeze(0) for k, v in entropy_maps[j].items()}
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        _ = network(im.unsqueeze(0), em)
                    model_s += time.perf_counter() - t1

        e2e_s = time.perf_counter() - t_e2e_start

        if step >= args.warmup:
            total_images += bsz
            total_model_s += model_s
            total_e2e_s += e2e_s

        if (step + 1) % 20 == 0:
            phase = "measure" if step >= args.warmup else "warmup"
            print(
                f"[{phase}] step={step + 1}/{total_iters} "
                f"model_ms={model_s * 1000.0:.2f} e2e_ms={e2e_s * 1000.0:.2f}"
            )

    model_fps = total_images / max(total_model_s, 1e-9)
    e2e_fps = total_images / max(total_e2e_s, 1e-9)
    mean_model_ms = (total_model_s / max(args.iters, 1)) * 1000.0
    mean_e2e_ms = (total_e2e_s / max(args.iters, 1)) * 1000.0

    print("\n=== FPS Summary ===")
    print(f"images_processed: {total_images}")
    print(f"model_only_fps: {model_fps:.3f}  (mean_model_ms={mean_model_ms:.3f})")
    print(f"end_to_end_fps: {e2e_fps:.3f}  (mean_e2e_ms={mean_e2e_ms:.3f})")


if __name__ == "__main__":
    main()
