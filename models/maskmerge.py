# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# MaskMerge implementation adapted for EoMT-style ViT segmentation.
# ---------------------------------------------------------------

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scale_block import ScaleBlock


def _batched_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _l1_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def _js_similarity(
    a_probs: torch.Tensor,
    b_probs: torch.Tensor,
    eps: float = 1e-12,
    chunk_size: int = 128,
) -> torch.Tensor:
    a_probs = _l1_normalize(a_probs, eps=eps).clamp_min(eps)
    b_probs = _l1_normalize(b_probs, eps=eps).clamp_min(eps)
    a_log = a_probs.log()
    b_log = b_probs.log()
    out = a_probs.new_empty(a_probs.shape[0], a_probs.shape[1], b_probs.shape[1])
    log2 = math.log(2.0)

    for start in range(0, a_probs.shape[1], chunk_size):
        end = min(start + chunk_size, a_probs.shape[1])
        a = a_probs[:, start:end].unsqueeze(2)
        b = b_probs.unsqueeze(1)
        m = (a + b) * 0.5
        m_log = m.clamp_min(eps).log()
        kl_am = (a * (a_log[:, start:end].unsqueeze(2) - m_log)).sum(dim=-1)
        kl_bm = (b * (b_log.unsqueeze(1) - m_log)).sum(dim=-1)
        out[:, start:end] = 1.0 - (0.5 * (kl_am + kl_bm) / log2)

    return out


def _maskmerge_scores(
    metric: torch.Tensor,
    mask_probs: torch.Tensor,
    score_metric: str,
) -> torch.Tensor:
    eps = 1e-12
    if score_metric == "cosine":
        a = metric[:, ::2]
        b = metric[:, 1::2]
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
        return a @ b.transpose(-1, -2)

    a_probs = mask_probs[:, :, ::2].transpose(1, 2)
    b_probs = mask_probs[:, :, 1::2].transpose(1, 2)
    if score_metric == "js":
        return _js_similarity(a_probs, b_probs, eps=eps)
    if score_metric == "hellinger":
        a = _l1_normalize(a_probs, eps=eps).clamp_min(eps).sqrt()
        b = _l1_normalize(b_probs, eps=eps).clamp_min(eps).sqrt()
        return a @ b.transpose(-1, -2)
    if score_metric == "dot":
        a = _l1_normalize(a_probs, eps=eps)
        b = _l1_normalize(b_probs, eps=eps)
        return a @ b.transpose(-1, -2)
    raise ValueError("global_merge_score_metric must be one of: cosine, js, hellinger, dot")


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = False,
        local_merge_enabled: bool = True,
        local_merge_after_layer: int = 1,
        local_merge_window_size: tuple[int, int] = (2, 2),
        local_merge_threshold: float = 0.88,
        merge_start_layer: int = 7,
        merge_interval: int = 20,
        merge_conf_threshold: float = 0.7,
        merge_conf_threshold_end: float = 0.0,
        merge_schedule_epochs: int = 7,
        merge_schedule_enabled: bool = True,
        global_merge_score_metric: str = "cosine",
        global_merge_min_ratio: float = 0.0,
        global_merge_mask_constraint: bool = True,
        global_merge_threshold_mode: str = "algm",
        olam_mask_source: str = "current",
        merge_inside_block: bool = True,
        query_insert_layer: Optional[int] = 7,
        query_reinsert_layer: Optional[int] = 20,
        merge_logging_enabled: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = int(num_q)
        self.num_blocks = int(num_blocks)
        self.masked_attn_enabled = bool(masked_attn_enabled)
        self.local_merge_enabled = bool(local_merge_enabled)
        self.local_merge_after_layer = int(local_merge_after_layer)
        self.local_merge_window_size = tuple(local_merge_window_size)
        self.local_merge_threshold = float(local_merge_threshold)
        self.merge_start_layer = int(merge_start_layer)
        self.merge_interval = int(merge_interval)
        self.merge_conf_threshold = float(merge_conf_threshold)
        self.merge_conf_threshold_start = float(merge_conf_threshold)
        self.merge_conf_threshold_end = float(merge_conf_threshold_end)
        self.merge_schedule_epochs = int(merge_schedule_epochs)
        self.merge_schedule_enabled = bool(merge_schedule_enabled)
        self.global_merge_score_metric = str(global_merge_score_metric)
        self.global_merge_min_ratio = float(global_merge_min_ratio)
        self.global_merge_mask_constraint = bool(global_merge_mask_constraint)
        self.global_merge_threshold_mode = str(global_merge_threshold_mode)
        self.olam_mask_source = str(olam_mask_source)
        self.merge_inside_block = bool(merge_inside_block)
        self.query_insert_layer = query_insert_layer
        self.query_reinsert_layer = query_reinsert_layer
        self.merge_logging_enabled = bool(merge_logging_enabled)

        if self.global_merge_score_metric not in ("cosine", "js", "hellinger", "dot"):
            raise ValueError(
                "global_merge_score_metric must be one of: 'cosine', 'js', 'hellinger', 'dot'"
            )
        if self.olam_mask_source not in ("current", "full"):
            raise ValueError("olam_mask_source must be either 'current' or 'full'")
        if not 0.0 <= self.global_merge_min_ratio <= 1.0:
            raise ValueError("global_merge_min_ratio must be in [0, 1]")
        if self.global_merge_threshold_mode not in ("algm", "per_image"):
            raise ValueError("global_merge_threshold_mode must be either 'algm' or 'per_image'")

        self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))
        self.q = nn.Embedding(self.num_q, self.encoder.backbone.embed_dim)
        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)]
        )

    @property
    def _num_prefix_tokens(self) -> int:
        return int(self.encoder.backbone.num_prefix_tokens)

    @property
    def _grid_size(self) -> tuple[int, int]:
        grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
        return int(grid_h), int(grid_w)

    def update_merge_schedule(self, epoch: int, max_epochs: Optional[int] = None) -> None:
        if not self.merge_schedule_enabled:
            return
        total_epochs = max(int(self.merge_schedule_epochs or max_epochs or 1), 1)
        if total_epochs == 1:
            t = 1.0
        else:
            t = min(max(int(epoch), 0), total_epochs - 1) / float(total_epochs - 1)
        start = self.merge_conf_threshold_start
        end = self.merge_conf_threshold_end
        self.merge_conf_threshold = start + t * (end - start)

    def _resolve_layer(self, layer: Optional[int]) -> int:
        if layer is None:
            return len(self.encoder.backbone.blocks) - self.num_blocks
        return min(max(int(layer) - 1, 0), len(self.encoder.backbone.blocks) - 1)

    def _should_global_merge(self, layer_idx: int) -> bool:
        if self.merge_interval <= 0:
            return False
        if layer_idx < self.merge_start_layer:
            return False
        return (layer_idx - self.merge_start_layer) % self.merge_interval == 0

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = block(x)
        return out[0] if isinstance(out, tuple) else out

    def _can_split_block(self, block: nn.Module) -> bool:
        return all(hasattr(block, attr) for attr in ("norm1", "attn", "norm2", "mlp"))

    def _drop_path1(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "drop_path1"):
            return block.drop_path1(x)
        if hasattr(block, "drop_path"):
            return block.drop_path(x)
        return x

    def _drop_path2(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "drop_path2"):
            return block.drop_path2(x)
        if hasattr(block, "drop_path"):
            return block.drop_path(x)
        return x

    def _scale1(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "ls1"):
            return block.ls1(x)
        if hasattr(block, "layer_scale1"):
            return block.layer_scale1(x)
        return x

    def _scale2(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "ls2"):
            return block.ls2(x)
        if hasattr(block, "layer_scale2"):
            return block.layer_scale2(x)
        return x

    def _run_block_attention(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = block.attn(block.norm1(x))
        attn_out = out[0] if isinstance(out, tuple) else out
        return x + self._drop_path1(block, self._scale1(block, attn_out))

    # --- masked-attention annealing (EoMT decoder trick) over MERGED tokens ---
    # Standard EoMT masks each query's attention to the patches inside its
    # predicted mask, then anneals that constraint away over training. dense
    # builds the mask on a spatial grid; here the decoder tokens are already
    # MERGED (no grid), so the mask is built directly from a query.vision dot
    # product -- the same signal MaskMerge already uses to gate merges
    # (_token_mask_probs 'current'). Everything below only runs when
    # masked_attn_enabled is True; the baseline (False) path is untouched.
    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask: torch.Tensor, prob) -> torch.Tensor:
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self._num_prefix_tokens :
            ][random_queries] = True
        return attn_mask

    def _coarse_attn_mask(self, x_norm: torch.Tensor, block_pos: int) -> torch.Tensor:
        protected = self.num_q + self._num_prefix_tokens
        q = x_norm[:, : self.num_q]
        vision = x_norm[:, protected:]
        logits = torch.einsum("bqc,bnc->bqn", self.mask_head(q), vision)
        attn_mask = torch.ones(
            x_norm.shape[0], x_norm.shape[1], x_norm.shape[1],
            dtype=torch.bool, device=x_norm.device,
        )
        attn_mask[:, : self.num_q, protected:] = logits > 0
        return self._disable_attn_mask(attn_mask, self.attn_mask_probs[block_pos])

    def _attn_with_mask(
        self, block: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        module = block.attn
        bsz, seq, dim = x.shape
        qkv = module.qkv(x).reshape(bsz, seq, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask)
        return module.proj_drop(module.proj(out.transpose(1, 2).reshape(bsz, seq, dim)))

    def _run_block_attention_masked(
        self, block: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        attn_out = self._attn_with_mask(block, block.norm1(x), attn_mask)
        return x + self._drop_path1(block, self._scale1(block, attn_out))

    def _run_block_mlp(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        mlp_out = block.mlp(block.norm2(x))
        return x + self._drop_path2(block, self._scale2(block, mlp_out))

    def _prepend_queries(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q.weight[None].expand(x.shape[0], -1, -1)
        return torch.cat((q, x), dim=1)

    def _predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = x[:, : self.num_q]
        class_logits = self.class_head(q)

        x = x[:, self.num_q + self._num_prefix_tokens :]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self._grid_size)
        mask_logits = torch.einsum("bqc,bchw->bqhw", self.mask_head(q), self.upscale(x))
        return mask_logits, class_logits

    def _token_mask_probs(
        self,
        x_norm: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool,
    ) -> torch.Tensor:
        if not has_queries:
            raise ValueError("OLAM requires query tokens to compute coarse masks.")

        protected = self.num_q + self._num_prefix_tokens
        if self.olam_mask_source == "current":
            q = x_norm[:, : self.num_q]
            vision = x_norm[:, protected:]
            logits = torch.einsum("bqc,bnc->bqn", self.mask_head(q), vision)
            return torch.softmax(logits, dim=1)

        x_full = self._unmerge_to_base(x_norm, base_to_current, has_queries=True)
        mask_logits, _ = self._predict(x_full)
        mask_logits = F.interpolate(mask_logits, self._grid_size, mode="bilinear")
        base_probs = torch.softmax(mask_logits.flatten(2), dim=1)
        return self._project_base_probs(base_probs, base_to_current, x_norm.shape[1] - protected)

    def _project_base_probs(
        self,
        base_probs: torch.Tensor,
        base_to_current: torch.Tensor,
        num_current: int,
    ) -> torch.Tensor:
        bsz, num_q, num_base = base_probs.shape
        if base_to_current.shape != (bsz, num_base):
            raise ValueError(
                f"base_to_current shape {tuple(base_to_current.shape)} does not match "
                f"base probs {(bsz, num_base)}"
            )
        out = base_probs.new_zeros(bsz, num_q, num_current)
        idx = base_to_current.unsqueeze(1).expand(-1, num_q, -1)
        out.scatter_add_(2, idx, base_probs)
        counts = base_probs.new_zeros(bsz, num_current)
        counts.scatter_add_(1, base_to_current, torch.ones_like(base_to_current, dtype=base_probs.dtype))
        return out / counts.clamp_min(1.0).unsqueeze(1)

    def _unmerge_to_base(
        self,
        x: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool,
    ) -> torch.Tensor:
        protected = self._num_prefix_tokens + (self.num_q if has_queries else 0)
        vision = x[:, protected:]
        if vision.shape[1] == base_to_current.shape[1] and torch.equal(
            base_to_current,
            torch.arange(base_to_current.shape[1], device=base_to_current.device)
            .unsqueeze(0)
            .expand_as(base_to_current),
        ):
            return x
        if int(base_to_current.max().item()) >= vision.shape[1]:
            raise ValueError(
                f"base_to_current index out of range: max={int(base_to_current.max().item())}, "
                f"vision_tokens={vision.shape[1]}"
            )
        vision_full = _batched_gather(vision, base_to_current)
        return torch.cat((x[:, :protected], vision_full), dim=1)

    def _local_merge(
        self,
        vision: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        bsz, num_tokens, dim = vision.shape
        grid_h, grid_w = self._grid_size
        win_h, win_w = int(self.local_merge_window_size[0]), int(self.local_merge_window_size[1])
        if win_h <= 0 or win_w <= 0:
            return vision, sizes, base_to_current, False
        window_tokens = win_h * win_w
        core_h = (grid_h // win_h) * win_h
        core_w = (grid_w // win_w) * win_w
        if (
            num_tokens != grid_h * grid_w
            or core_h == 0
            or core_w == 0
        ):
            return vision, sizes, base_to_current, False

        token_grid = torch.arange(num_tokens, device=vision.device).reshape(grid_h, grid_w)
        window_ids = (
            token_grid[:core_h, :core_w]
            .reshape(core_h // win_h, win_h, core_w // win_w, win_w)
            .permute(0, 2, 1, 3)
            .reshape(-1, window_tokens)
        )
        windows = _batched_gather(
            vision,
            window_ids.reshape(1, -1).expand(bsz, -1),
        ).reshape(bsz, window_ids.shape[0], window_tokens, dim)
        normalized = windows / windows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sims = normalized @ normalized.transpose(-1, -2)
        sims = sims - torch.eye(window_tokens, device=vision.device, dtype=vision.dtype)
        mean_sim = sims.sum(dim=(-1, -2)) / float(window_tokens * (window_tokens - 1))
        eligible = mean_sim >= self.local_merge_threshold
        r = int(eligible.sum(dim=1).min().item())
        if r <= 0:
            return vision, sizes, base_to_current, False

        selected_windows = mean_sim.masked_fill(~eligible, -torch.inf).topk(r, dim=1).indices
        selected_ids = window_ids[selected_windows.reshape(-1)].reshape(bsz, r, window_tokens)

        selected_flat = selected_ids.reshape(bsz, -1)
        selected_mask = torch.zeros(bsz, num_tokens, device=vision.device, dtype=torch.bool)
        selected_mask.scatter_(1, selected_flat, True)

        all_ids = torch.arange(num_tokens, device=vision.device).unsqueeze(0).expand(bsz, -1)
        remaining_ids = all_ids[~selected_mask].reshape(bsz, num_tokens - r * window_tokens)

        selected_tokens = _batched_gather(vision, selected_flat).reshape(bsz, r, window_tokens, dim)
        selected_sizes = _batched_gather(sizes, selected_flat).reshape(bsz, r, window_tokens, 1)
        merged_sizes = selected_sizes.sum(dim=2).clamp_min(1e-12)
        merged_tokens = (selected_tokens * selected_sizes).sum(dim=2) / merged_sizes

        remaining_tokens = _batched_gather(vision, remaining_ids)
        remaining_sizes = _batched_gather(sizes, remaining_ids)
        new_vision = torch.cat((merged_tokens, remaining_tokens), dim=1)
        new_sizes = torch.cat((merged_sizes, remaining_sizes), dim=1)

        old_to_new = torch.empty(bsz, num_tokens, device=vision.device, dtype=torch.long)
        old_to_new.scatter_(
            1,
            selected_flat,
            torch.arange(r, device=vision.device).repeat_interleave(window_tokens).unsqueeze(0).expand(bsz, -1),
        )
        old_to_new.scatter_(
            1,
            remaining_ids,
            torch.arange(remaining_ids.shape[1], device=vision.device).unsqueeze(0).expand(bsz, -1) + r,
        )
        new_base_to_current = old_to_new.gather(1, base_to_current)
        return new_vision, new_sizes, new_base_to_current, True

    def _global_merge(
        self,
        vision: torch.Tensor,
        sizes: torch.Tensor,
        mask_probs: torch.Tensor,
        base_to_current: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        bsz, num_tokens, dim = vision.shape
        dst_len = num_tokens // 2
        src_len = (num_tokens + 1) // 2
        if dst_len == 0:
            return vision, sizes, base_to_current, False

        with torch.no_grad():
            conf, mask_id = mask_probs.max(dim=1)
            scores = _maskmerge_scores(vision, mask_probs, self.global_merge_score_metric)

            a_conf = conf[:, ::2]
            b_conf = conf[:, 1::2]
            ok = (a_conf >= self.merge_conf_threshold).unsqueeze(-1) & (
                b_conf >= self.merge_conf_threshold
            ).unsqueeze(-2)
            if self.global_merge_mask_constraint:
                ok = ok & (mask_id[:, ::2].unsqueeze(-1) == mask_id[:, 1::2].unsqueeze(-2))
            scores = scores.masked_fill(~ok, -torch.inf)

            node_max, node_idx = scores.max(dim=-1)
            valid = torch.isfinite(node_max)
            if not valid.any():
                return vision, sizes, base_to_current, False

            valid_counts = valid.sum(dim=1).clamp(min=1)
            masked = node_max.masked_fill(~valid, 0.0)
            per_image_mean = masked.sum(dim=1) / valid_counts
            per_image_var = (
                ((masked - per_image_mean.unsqueeze(-1)) ** 2 * valid).sum(dim=1)
                / valid_counts
            )
            per_image_std = per_image_var.sqrt()
            if self.global_merge_threshold_mode == "algm":
                threshold = per_image_mean.mean() + per_image_std.mean() / max(int(layer_idx), 1)
                adaptive_r = ((node_max >= threshold) & valid).sum(dim=1).min()
            else:
                threshold = per_image_mean + per_image_std / max(int(layer_idx), 1)
                adaptive_r = ((node_max >= threshold.unsqueeze(-1)) & valid).sum(dim=1).min()
            max_valid = int(valid_counts.min().item())
            ratio_r = int(max_valid * self.global_merge_min_ratio)
            r = min(max(int(adaptive_r.item()), ratio_r), max_valid, src_len)
            if r <= 0:
                return vision, sizes, base_to_current, False

            edge_idx = node_max.argsort(dim=1, descending=True).unsqueeze(-1)
            src_idx = edge_idx[:, :r]
            unm_idx = edge_idx[:, r:]
            dst_idx = node_idx.unsqueeze(-1).gather(1, src_idx)

        src_tokens = vision[:, ::2]
        dst_tokens = vision[:, 1::2]
        src_sizes = sizes[:, ::2]
        dst_sizes = sizes[:, 1::2]

        unm_tokens = _batched_gather(src_tokens, unm_idx.squeeze(-1))
        unm_sizes = _batched_gather(src_sizes, unm_idx.squeeze(-1))
        picked_src = _batched_gather(src_tokens, src_idx.squeeze(-1))
        picked_src_sizes = _batched_gather(src_sizes, src_idx.squeeze(-1))

        dst_sum = dst_tokens * dst_sizes
        dst_size_sum = dst_sizes.clone()
        dst_index = dst_idx.expand(-1, -1, dim)
        dst_sum.scatter_add_(1, dst_index, picked_src * picked_src_sizes)
        dst_size_sum.scatter_add_(1, dst_idx, picked_src_sizes)
        merged_dst = dst_sum / dst_size_sum.clamp_min(1e-12)

        new_vision = torch.cat((unm_tokens, merged_dst), dim=1)
        new_sizes = torch.cat((unm_sizes, dst_size_sum), dim=1)

        old_to_new = torch.empty(bsz, num_tokens, device=vision.device, dtype=torch.long)
        old_to_new[:, 1::2] = torch.arange(dst_len, device=vision.device).unsqueeze(0) + (src_len - r)
        if unm_idx.numel() > 0:
            old_to_new.scatter_(
                1,
                2 * unm_idx.squeeze(-1),
                torch.arange(src_len - r, device=vision.device).unsqueeze(0).expand(bsz, -1),
            )
        old_to_new.scatter_(1, 2 * src_idx.squeeze(-1), dst_idx.squeeze(-1) + (src_len - r))
        new_base_to_current = old_to_new.gather(1, base_to_current)
        return new_vision, new_sizes, new_base_to_current, True

    def _apply_local_merge(
        self,
        x: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        protected = self._num_prefix_tokens + (self.num_q if has_queries else 0)
        if (
            not self.local_merge_enabled
            or has_queries
            or layer_idx != max(self.local_merge_after_layer, 1)
        ):
            return x, sizes, base_to_current

        before = x.shape[1] - protected
        vision, sizes, base_to_current, did_merge = self._local_merge(
            x[:, protected:],
            sizes,
            base_to_current,
        )
        if did_merge:
            x = torch.cat((x[:, :protected], vision), dim=1)
            if self.merge_logging_enabled:
                warnings.warn(
                    f"[maskmerge:local] layer={layer_idx} vision={before}->{vision.shape[1]}"
                )
        return x, sizes, base_to_current

    def _apply_global_merge(
        self,
        x: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not has_queries or layer_idx >= len(self.encoder.backbone.blocks):
            return x, sizes, base_to_current
        if not self._should_global_merge(layer_idx):
            return x, sizes, base_to_current

        protected = self.num_q + self._num_prefix_tokens
        vision = x[:, protected:]
        before = vision.shape[1]
        probs = self._token_mask_probs(
            self.encoder.backbone.norm(x),
            base_to_current,
            has_queries=True,
        )
        vision, sizes, base_to_current, did_merge = self._global_merge(
            vision,
            sizes,
            probs,
            base_to_current,
            layer_idx=layer_idx,
        )
        if did_merge:
            x = torch.cat((x[:, :protected], vision), dim=1)
            if self.merge_logging_enabled:
                warnings.warn(
                    f"[maskmerge:olam] layer={layer_idx} tau={self.merge_conf_threshold:.3f} "
                    f"vision={before}->{vision.shape[1]}"
                )
        return x, sizes, base_to_current

    def _apply_selected_merges(
        self,
        x: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, sizes, base_to_current = self._apply_local_merge(
            x,
            sizes,
            base_to_current,
            has_queries,
            layer_idx,
        )
        x, sizes, base_to_current = self._apply_global_merge(
            x,
            sizes,
            base_to_current,
            has_queries,
            layer_idx,
        )
        return x, sizes, base_to_current

    def forward(
        self,
        x: torch.Tensor,
        entropy_maps: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        del entropy_maps

        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std
        ps = self.encoder.backbone.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else ps
        target_grid = (x.shape[-2] // ps, x.shape[-1] // ps)
        if not hasattr(self, "_creation_grid_size"):
            self._creation_grid_size = tuple(
                self.encoder.backbone.patch_embed.grid_size
            )
        # _pos_embed interpolates pos-embed using grid_size as the OLD size, so
        # restore the creation grid here (a prior forward may have left it at
        # that call's target), then set the ACTUAL input grid that _grid_size
        # (every merge/readout step below) reads. Enables eval at a resolution
        # other than the creation one.
        self.encoder.backbone.patch_embed.grid_size = self._creation_grid_size
        x = self.encoder.backbone.patch_embed(x)
        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)
        self.encoder.backbone.patch_embed.grid_size = target_grid

        bsz = x.shape[0]
        num_base = self._grid_size[0] * self._grid_size[1]
        base_to_current = torch.arange(num_base, device=x.device).unsqueeze(0).expand(bsz, -1)
        sizes = x.new_ones(bsz, num_base, 1)

        query_insert_layer = self._resolve_layer(self.query_insert_layer)
        query_reinsert_layer = self._resolve_layer(self.query_reinsert_layer)
        has_queries = False
        queries_persistent = False
        num_layers = len(self.encoder.backbone.blocks)
        # Deep-supervision predictions from each masked-attention decoder block
        # (only populated when masked_attn_enabled); the final prediction is
        # always appended below. Baseline path leaves these empty -> returns a
        # single-element list, identical to before.
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == query_insert_layer and not has_queries:
                x = self._prepend_queries(x)
                has_queries = True
                queries_persistent = i == query_reinsert_layer
            elif i == query_reinsert_layer and not has_queries:
                x = self._prepend_queries(x)
                has_queries = True
                queries_persistent = True

            layer_idx = i + 1
            wants_local = (
                self.local_merge_enabled
                and not has_queries
                and layer_idx == max(self.local_merge_after_layer, 1)
            )
            wants_global = (
                has_queries
                and layer_idx < len(self.encoder.backbone.blocks)
                and self._should_global_merge(layer_idx)
            )
            is_masked = (
                self.masked_attn_enabled
                and has_queries
                and i >= num_layers - self.num_blocks
                and self._can_split_block(block)
            )

            if is_masked:
                # Masked-attention decoder block over MERGED tokens. The mask
                # that gates this block's attention is built from x_norm; the
                # per-layer prediction is DEEP SUPERVISION only (its output is
                # unused at inference -- eval reads the final prediction after
                # the loop), so skip that unmerge+_predict when not training to
                # avoid ~4 wasted full-grid readouts per forward. Merging never
                # fires here (global merge is at merge_start_layer only), so the
                # merged-tail is preserved.
                x_norm = self.encoder.backbone.norm(x)
                if self.training:
                    x_full = self._unmerge_to_base(x_norm, base_to_current, has_queries=True)
                    ml, cl = self._predict(x_full)
                    mask_logits_per_layer.append(ml)
                    class_logits_per_layer.append(cl)
                attn_mask = self._coarse_attn_mask(x_norm, i - (num_layers - self.num_blocks))
                x = self._run_block_attention_masked(block, x, attn_mask)
                x, sizes, base_to_current = self._apply_selected_merges(
                    x, sizes, base_to_current, has_queries, layer_idx,
                )
                x = self._run_block_mlp(block, x)
            elif self.merge_inside_block and (wants_local or wants_global) and self._can_split_block(block):
                x = self._run_block_attention(block, x)
                x, sizes, base_to_current = self._apply_selected_merges(
                    x,
                    sizes,
                    base_to_current,
                    has_queries,
                    layer_idx,
                )
                x = self._run_block_mlp(block, x)
            else:
                x = self._run_block(block, x)
                x, sizes, base_to_current = self._apply_selected_merges(
                    x,
                    sizes,
                    base_to_current,
                    has_queries,
                    layer_idx,
                )

            if has_queries and not queries_persistent and i == query_insert_layer:
                x = x[:, self.num_q :]
                has_queries = False

        if not has_queries:
            x = self._prepend_queries(x)
            has_queries = True

        x = self._unmerge_to_base(x, base_to_current, has_queries=has_queries)
        if not torch.compiler.is_compiling():
            actual = x.shape[1] - self.num_q - self._num_prefix_tokens
            if actual != num_base:
                raise RuntimeError(f"MaskMerge unmerge produced {actual} vision tokens, expected {num_base}")

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits_per_layer, class_logits_per_layer
