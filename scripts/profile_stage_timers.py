#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import time
from contextlib import nullcontext
from collections import defaultdict
import copy
from pathlib import Path
from types import MethodType
from typing import Any
import sys

import torch
import yaml

# Ensure project root is importable when running script directly.
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


class StageProfiler:
    def __init__(self, use_cuda_timing: bool):
        self.use_cuda_timing = use_cuda_timing
        self.totals_ms = defaultdict(float)
        self.counts = defaultdict(int)
        self._originals = []
        self.enabled = True
        self.token_totals = defaultdict(float)
        self.token_counts = defaultdict(int)

    def _time_call(self, name: str, fn, *args, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        if self.use_cuda_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.totals_ms[name] += float(elapsed_ms)
        self.counts[name] += 1
        return out

    def add_timing(self, name: str, elapsed_ms: float):
        if not self.enabled:
            return
        self.totals_ms[name] += float(elapsed_ms)
        self.counts[name] += 1

    def wrap_method(self, obj, method_name: str, label: str | None = None):
        if not hasattr(obj, method_name):
            return
        label = label or method_name
        original = getattr(obj, method_name)
        if not callable(original):
            return

        def wrapped(this, *args, **kwargs):
            return self._time_call(label, original, *args, **kwargs)

        setattr(obj, method_name, MethodType(wrapped, obj))
        self._originals.append((obj, method_name, original))

    def restore(self):
        for obj, method_name, original in reversed(self._originals):
            setattr(obj, method_name, original)
        self._originals.clear()

    def record_token_stat(self, name: str, value: float, n: int = 1):
        self.token_totals[name] += float(value)
        self.token_counts[name] += int(n)

    def report(self, total_model_ms: float, total_model_calls: int, total_data_ms: float, total_data_calls: int):
        print("\n=== Timing Summary ===")
        if total_data_calls > 0:
            print(f"dataloader_wait_ms: avg={total_data_ms / total_data_calls:.3f} over {total_data_calls} batches")
        if total_model_calls > 0:
            avg_model_ms = total_model_ms / total_model_calls
            print(f"model_forward_ms: avg={avg_model_ms:.3f} over {total_model_calls} iters")

        if not self.totals_ms:
            print("No stage timers recorded.")
            return

        print("\n--- Stage Breakdown (inside model) ---")
        for name in sorted(self.totals_ms.keys(), key=lambda k: self.totals_ms[k], reverse=True):
            total = self.totals_ms[name]
            cnt = self.counts[name]
            avg = total / max(cnt, 1)
            pct = (total / total_model_ms * 100.0) if total_model_ms > 0 else 0.0
            print(f"{name:40s} avg={avg:8.3f} ms  calls={cnt:5d}  share={pct:6.2f}%")

        if self.token_totals:
            print("\n--- Token Stats ---")
            for name in sorted(self.token_totals.keys()):
                total = self.token_totals[name]
                cnt = self.token_counts[name]
                avg = total / max(cnt, 1)
                print(f"{name:40s} avg={avg:8.3f}  count={cnt:5d}")
            if "entropy_pre_vision_tokens" in self.token_totals and "global_post_vision_tokens" in self.token_totals:
                pre = self.token_totals["entropy_pre_vision_tokens"] / max(self.token_counts["entropy_pre_vision_tokens"], 1)
                post = self.token_totals["global_post_vision_tokens"] / max(self.token_counts["global_post_vision_tokens"], 1)
                if pre > 0:
                    red = (1.0 - (post / pre)) * 100.0
                    print(f"{'global_merge_reduction_percent':40s} avg={red:8.3f}%")


def _to_device_batch(batch, device: torch.device):
    if len(batch) == 3:
        imgs, targets, entropy_maps = batch
    else:
        imgs, targets = batch
        entropy_maps = None

    imgs = imgs.to(device, non_blocking=True)
    if entropy_maps is not None:
        entropy_maps = {k: v.to(device, non_blocking=True) for k, v in entropy_maps.items()}
    return imgs, targets, entropy_maps


def main():
    parser = argparse.ArgumentParser(description="Standalone stage timer for EoMT forward.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/token_merge/stage3/panoptic_finetune_large_640.yaml"),
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader num_workers")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader batch_size")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--data-path", type=str, default=None, help="Override data.init_args.path")
    parser.add_argument(
        "--amp",
        type=str,
        default="bf16",
        choices=["off", "bf16", "fp16"],
        help="Autocast dtype for CUDA forward timing.",
    )
    parser.add_argument("--train-mode", action="store_true", help="Run model in train() mode instead of eval().")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    network_cfg = copy.deepcopy(cfg["model"]["init_args"]["network"])
    data_cfg = copy.deepcopy(cfg["data"])

    data_init = data_cfg.setdefault("init_args", {})
    if args.data_path is not None:
        data_init["path"] = args.data_path
    if "path" not in data_init:
        raise ValueError(
            "Missing data path. Set data.init_args.path in config or pass --data-path."
        )

    if args.img_size is not None:
        img_size = tuple(args.img_size)
    else:
        img_size = tuple(data_init.get("img_size", (640, 640)))
    data_init["img_size"] = img_size

    # Lightning config often relies on linked args. Fill encoder img_size for standalone use.
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
                print(f"Inferred num_classes={param.default} from {data_cfg['class_path']}.__init__ default")
            else:
                raise ValueError(
                    "Missing network num_classes. Set model.init_args.network.init_args.num_classes "
                    "or data.init_args.num_classes in config."
                )
    encoder_cfg = net_init.get("encoder")
    if isinstance(encoder_cfg, dict) and "class_path" in encoder_cfg:
        enc_init = encoder_cfg.setdefault("init_args", {})
        enc_init.setdefault("img_size", img_size)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    print(f"Loading network from: {network_cfg['class_path']}")
    network = _instantiate(network_cfg).to(device)
    if args.train_mode:
        network.train()
    else:
        network.eval()

    print(f"Loading datamodule from: {data_cfg['class_path']}")
    dm = _instantiate(data_cfg)
    if args.num_workers is not None:
        dm.dataloader_kwargs["num_workers"] = args.num_workers
        dm.dataloader_kwargs["persistent_workers"] = args.num_workers > 0
    if args.batch_size is not None:
        dm.dataloader_kwargs["batch_size"] = args.batch_size
    dm.setup("fit")
    loader = dm.train_dataloader()

    profiler = StageProfiler(use_cuda_timing=(device.type == "cuda"))
    # Core stages for entropy + varlen + global merge path.
    for name in [
        "_compute_entropy_maps",
        "_compute_entropy_keep_mask_from_maps",
        "_compute_entropy_keep_mask",
        "_entropy_pre_transformer_tokenize_varlen",
        "_merge_tokens",
        "_predict",
    ]:
        profiler.wrap_method(network, name)

    # Token accounting hooks for varlen entropy + global merge path.
    if hasattr(network, "_entropy_pre_transformer_tokenize_varlen"):
        original_entropy_tokenize = network._entropy_pre_transformer_tokenize_varlen

        def wrapped_entropy_tokenize(this, *a, **kw):
            token_list, source_vis_list = original_entropy_tokenize(*a, **kw)
            if profiler.enabled and token_list:
                num_prefix = int(this.encoder.backbone.num_prefix_tokens)
                vis_sum = 0
                for t in token_list:
                    vis_sum += max(int(t.shape[0]) - num_prefix, 0)
                profiler.record_token_stat("entropy_pre_vision_tokens", vis_sum, n=len(token_list))
            return token_list, source_vis_list

        network._entropy_pre_transformer_tokenize_varlen = MethodType(wrapped_entropy_tokenize, network)
        profiler._originals.append((network, "_entropy_pre_transformer_tokenize_varlen", original_entropy_tokenize))

    try:
        import models.eomt_early_layer as eomt_layer_mod

        original_global_match = eomt_layer_mod.global_bipartite_matching_masked

        def wrapped_global_match(x, mask_probs, *a, **kw):
            merge, merge_info = original_global_match(x, mask_probs, *a, **kw)
            if profiler.enabled:
                old_len = int(x.shape[1])
                r = 0
                if merge_info is not None and "src_idx" in merge_info:
                    r = int(merge_info["src_idx"].shape[1])
                new_len = max(old_len - r, 0)
                profiler.record_token_stat("global_pre_vision_tokens", old_len)
                profiler.record_token_stat("global_post_vision_tokens", new_len)
            return merge, merge_info

        eomt_layer_mod.global_bipartite_matching_masked = wrapped_global_match
        profiler._originals.append((eomt_layer_mod, "global_bipartite_matching_masked", original_global_match))
    except Exception as ex:
        print(f"Warning: could not attach global token hook: {ex}")

    # Per-layer varlen block timing + token stats.
    if hasattr(network, "_run_block_varlen") and hasattr(network.encoder.backbone, "blocks"):
        block_to_idx = {id(b): i for i, b in enumerate(network.encoder.backbone.blocks)}
        original_run_block = network._run_block_varlen

        def wrapped_run_block(this, block, packed_tokens, cu_seqlens, max_seqlen, seqlens):
            layer_idx = block_to_idx.get(id(block), -1)
            layer_tag = f"layer_{layer_idx + 1:02d}" if layer_idx >= 0 else "layer_unknown"

            if profiler.enabled:
                # Avg sequence length entering this layer.
                profiler.record_token_stat(f"{layer_tag}_seq_tokens", float(seqlens.float().mean().item()))
                # Approx vision tokens by subtracting backbone prefix only.
                num_prefix = int(this.encoder.backbone.num_prefix_tokens)
                est_vis = (seqlens - num_prefix).clamp_min(0).float().mean().item()
                profiler.record_token_stat(f"{layer_tag}_vision_tokens_est", float(est_vis))

            if profiler.use_cuda_timing and packed_tokens.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = original_run_block(block, packed_tokens, cu_seqlens, max_seqlen, seqlens)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                out = original_run_block(block, packed_tokens, cu_seqlens, max_seqlen, seqlens)
                elapsed_ms = float((time.perf_counter() - t0) * 1000.0)

            profiler.add_timing("_run_block_varlen", elapsed_ms)
            profiler.add_timing(f"_run_block_varlen/{layer_tag}", elapsed_ms)
            return out

        network._run_block_varlen = MethodType(wrapped_run_block, network)
        profiler._originals.append((network, "_run_block_varlen", original_run_block))

    total_model_ms = 0.0
    total_data_ms = 0.0
    total_model_calls = 0
    total_data_calls = 0

    it = iter(loader)
    total_iters = args.warmup + args.iters
    print(
        f"Running warmup={args.warmup}, measured={args.iters}, "
        f"batch_size={dm.dataloader_kwargs['batch_size']}, device={device}"
    )

    if device.type == "cuda" and args.amp != "off":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext

    for step in range(total_iters):
        profiler.enabled = step >= args.warmup
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        data_ms = (time.perf_counter() - t0) * 1000.0

        imgs, _, entropy_maps = _to_device_batch(batch, device)

        if step >= args.warmup:
            total_data_ms += data_ms
            total_data_calls += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.set_grad_enabled(args.train_mode):
                with amp_ctx():
                    _ = network(imgs, entropy_maps)
            end.record()
            torch.cuda.synchronize()
            model_ms = start.elapsed_time(end)
        else:
            t1 = time.perf_counter()
            with torch.set_grad_enabled(args.train_mode):
                _ = network(imgs, entropy_maps)
            model_ms = (time.perf_counter() - t1) * 1000.0

        if step >= args.warmup:
            total_model_ms += float(model_ms)
            total_model_calls += 1

        if (step + 1) % 10 == 0:
            phase = "measure" if step >= args.warmup else "warmup"
            print(f"[{phase}] step={step + 1}/{total_iters} model_ms={model_ms:.3f}")

    profiler.report(
        total_model_ms=total_model_ms,
        total_model_calls=total_model_calls,
        total_data_ms=total_data_ms,
        total_data_calls=total_data_calls,
    )
    profiler.restore()


if __name__ == "__main__":
    main()
