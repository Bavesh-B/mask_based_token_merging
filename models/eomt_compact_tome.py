# ---------------------------------------------------------------
# ToMe (global bipartite cosine merging) for EoMT with compact O(N) bookkeeping.
# Copy of eomt_compact_algm with the local 2x2 pre-merge disabled (merge_started
# starts True), so every selected layer runs the pure global bipartite cosine merge.
# This is the ToMe cosine baseline: appearance-only, no locality prior, no mask gate.
#
# Merge decisions and token ordering match models.eomt_original_algm.  The
# difference is bookkeeping: base_to_current[b, i] stores the current token
# that represents original patch i.  This replaces ALGM's dense O(N^2)
# source matrix and makes unmerging a single gather.
# ---------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import torch
import torch.nn as nn

from models.eomt_ori import EoMT as BaseEoMT


def _batched_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather token rows independently for each batch item."""
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


@dataclass
class BipartitePlan:
    """Indices required by one even-to-odd bipartite merge."""

    src_idx: torch.Tensor
    dst_idx: torch.Tensor
    unm_idx: torch.Tensor

    @property
    def num_merges(self) -> int:
        return int(self.src_idx.shape[1])


class EoMT(BaseEoMT):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        algm_selected_layers: Optional[list[int]] = None,
        local_merge_threshold: float = 0.95,
        local_merge_window_size: tuple[int, int] = (2, 2),
        merge_logging_enabled: bool = False,
        global_threshold_scale: float = 1.0,
        global_merge_ratio: float = 0.0,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
        )
        self.algm_selected_layers = [
            int(layer) for layer in (algm_selected_layers or [1, 7])
        ]
        self.local_merge_threshold = float(local_merge_threshold)
        self.local_merge_window_size = tuple(local_merge_window_size)
        self.merge_logging_enabled = bool(merge_logging_enabled)
        # ToMe aggressiveness knob: scale the adaptive bipartite threshold so ToMe can
        # be pushed to a matched token budget (<1 -> lower threshold -> more merges).
        self.global_threshold_scale = float(global_threshold_scale)
        # Fixed-ratio (canonical ToMe): if >0, merge a FIXED fraction of the src tokens
        # per layer (the r most-similar bipartite pairs), ignoring the adaptive threshold.
        # This lets ToMe reach ANY budget, so we can trace the full accuracy-FLOPs curve.
        self.global_merge_ratio = float(global_merge_ratio)

    @property
    def _num_prefix_tokens(self) -> int:
        return int(self.encoder.backbone.num_prefix_tokens)

    @property
    def _grid_size(self) -> tuple[int, int]:
        grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
        return int(grid_h), int(grid_w)

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

    @staticmethod
    def _scale1(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "ls1"):
            return block.ls1(x)
        if hasattr(block, "layer_scale1"):
            return block.layer_scale1(x)
        return x

    @staticmethod
    def _scale2(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "ls2"):
            return block.ls2(x)
        if hasattr(block, "layer_scale2"):
            return block.layer_scale2(x)
        return x

    def _run_block_attention(
        self,
        block: nn.Module,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn = block.attn if hasattr(block, "attn") else block.attention
        attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
        return x + self._drop_path1(block, self._scale1(block, attn_out))

    def _run_block_mlp(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        mlp_out = block.mlp(block.norm2(x))
        return x + self._drop_path2(block, self._scale2(block, mlp_out))

    def _initial_merge_state(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        num_base = self._grid_size[0] * self._grid_size[1]
        base_to_current = (
            torch.arange(num_base, device=x.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        sizes = x.new_ones(batch, num_base, 1)
        return sizes, base_to_current

    def _local_merge(
        self,
        vision: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Merge complete local windows selected by ALGM's CLAP score."""
        batch, num_tokens, dim = vision.shape
        grid_h, grid_w = self._grid_size
        win_h, win_w = (int(v) for v in self.local_merge_window_size)
        window_tokens = win_h * win_w

        if (
            win_h <= 0
            or win_w <= 0
            or num_tokens != grid_h * grid_w
            or grid_h % win_h
            or grid_w % win_w
        ):
            return vision, sizes, base_to_current, False

        token_grid = torch.arange(num_tokens, device=vision.device).reshape(
            grid_h, grid_w
        )
        window_ids = (
            token_grid.reshape(
                grid_h // win_h,
                win_h,
                grid_w // win_w,
                win_w,
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, window_tokens)
        )

        flat_ids = window_ids.flatten().unsqueeze(0).expand(batch, -1)
        windows = _batched_gather(vision, flat_ids).reshape(
            batch, -1, window_tokens, dim
        )
        normalized = windows / windows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarities = normalized @ normalized.transpose(-1, -2)
        similarities = similarities - torch.eye(
            window_tokens,
            device=vision.device,
            dtype=vision.dtype,
        )
        mean_similarity = similarities.sum(dim=(-1, -2)) / float(
            window_tokens * (window_tokens - 1)
        )

        eligible = mean_similarity >= self.local_merge_threshold
        num_windows = int(eligible.sum(dim=1).min().item())
        if num_windows <= 0:
            return vision, sizes, base_to_current, False

        selected_windows = mean_similarity.topk(num_windows, dim=1).indices
        selected_ids = window_ids[selected_windows.reshape(-1)].reshape(
            batch, num_windows, window_tokens
        )
        selected_flat = selected_ids.flatten(1)

        selected_mask = torch.zeros(
            batch,
            num_tokens,
            device=vision.device,
            dtype=torch.bool,
        )
        selected_mask.scatter_(1, selected_flat, True)
        all_ids = torch.arange(num_tokens, device=vision.device).unsqueeze(0)
        all_ids = all_ids.expand(batch, -1)
        remaining_ids = all_ids[~selected_mask].reshape(batch, -1)

        # Match ALGM's destination-plus-scatter reduction order exactly.  This
        # matters in mixed precision because tiny differences are amplified by
        # the following transformer blocks.
        src_ids = selected_ids[:, :, :-1].flatten(1)
        dst_ids = selected_ids[:, :, -1]
        merge_ids = (
            torch.arange(num_windows, device=vision.device)
            .repeat_interleave(window_tokens - 1)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        src_sum = _batched_gather(vision * sizes, src_ids)
        dst_sum = _batched_gather(vision * sizes, dst_ids)
        merged_sum = dst_sum.scatter_reduce(
            1,
            merge_ids.unsqueeze(-1).expand(-1, -1, dim),
            src_sum,
            reduce="sum",
        )
        src_sizes = _batched_gather(sizes, src_ids)
        dst_sizes = _batched_gather(sizes, dst_ids)
        merged_sizes = dst_sizes.scatter_reduce(
            1,
            merge_ids.unsqueeze(-1),
            src_sizes,
            reduce="sum",
        )
        merged_tokens = merged_sum / merged_sizes.clamp_min(1e-12)

        remaining_tokens = _batched_gather(vision, remaining_ids)
        remaining_sizes = _batched_gather(sizes, remaining_ids)
        new_vision = torch.cat((merged_tokens, remaining_tokens), dim=1)
        new_sizes = torch.cat((merged_sizes, remaining_sizes), dim=1)

        old_to_new = torch.empty(
            batch,
            num_tokens,
            device=vision.device,
            dtype=torch.long,
        )
        merged_ids = (
            torch.arange(num_windows, device=vision.device)
            .repeat_interleave(window_tokens)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        old_to_new.scatter_(1, selected_flat, merged_ids)
        remaining_new_ids = (
            torch.arange(remaining_ids.shape[1], device=vision.device)
            .unsqueeze(0)
            .expand(batch, -1)
            + num_windows
        )
        old_to_new.scatter_(1, remaining_ids, remaining_new_ids)
        base_to_current = old_to_new.gather(1, base_to_current)
        return new_vision, new_sizes, base_to_current, True

    def _global_plan(
        self,
        vision: torch.Tensor,
        layer_idx: int,
    ) -> Optional[BipartitePlan]:
        """Build the original adaptive ALGM global matching plan."""
        num_tokens = vision.shape[1]
        if num_tokens <= 1:
            return None

        with torch.no_grad():
            metric = vision / vision.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            src_metric = metric[:, ::2]
            dst_metric = metric[:, 1::2]
            if src_metric.shape[1] == 0 or dst_metric.shape[1] == 0:
                return None

            scores = src_metric @ dst_metric.transpose(-1, -2)
            node_max, node_idx = scores.max(dim=-1)
            threshold_scores = (
                node_max[:, 1:] if node_max.shape[1] > 1 else node_max
            )
            threshold = threshold_scores.mean(dim=1).mean()
            threshold = threshold + threshold_scores.std(
                dim=1, unbiased=False
            ).mean() / max(int(layer_idx), 1)
            threshold = threshold * self.global_threshold_scale
            if self.global_merge_ratio > 0.0:
                # canonical ToMe: merge a fixed fraction of src tokens (the top-r
                # most-similar bipartite pairs), regardless of the adaptive threshold.
                num_src = node_max.shape[1]
                num_merges = int(min(num_src, max(0, round(self.global_merge_ratio * num_src))))
            else:
                num_merges = int(
                    (node_max >= threshold).sum(dim=1).min().item()
                )
            if num_merges <= 0:
                return None

            edge_idx = node_max.argsort(dim=-1, descending=True)
            src_idx = edge_idx[:, :num_merges]
            unm_idx = edge_idx[:, num_merges:]
            dst_idx = node_idx.gather(1, src_idx)
            return BipartitePlan(src_idx, dst_idx, unm_idx)

    def _apply_bipartite_plan(
        self,
        vision: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
        plan: BipartitePlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a merge plan and compose the compact inverse map."""
        batch, num_tokens, dim = vision.shape
        num_merges = plan.num_merges
        src_tokens, dst_tokens = vision[:, ::2], vision[:, 1::2]
        src_sizes, dst_sizes = sizes[:, ::2], sizes[:, 1::2]

        unm_tokens = _batched_gather(src_tokens, plan.unm_idx)
        unm_sizes = _batched_gather(src_sizes, plan.unm_idx)
        picked_sizes = _batched_gather(src_sizes, plan.src_idx)

        weighted = vision * sizes
        weighted_src, weighted_dst = weighted[:, ::2], weighted[:, 1::2]
        picked_sum = _batched_gather(weighted_src, plan.src_idx)
        dst_sum = weighted_dst.scatter_reduce(
            1,
            plan.dst_idx.unsqueeze(-1).expand(-1, -1, dim),
            picked_sum,
            reduce="sum",
        )
        dst_size_sum = dst_sizes.scatter_reduce(
            1,
            plan.dst_idx.unsqueeze(-1),
            picked_sizes,
            reduce="sum",
        )
        dst_tokens = dst_sum / dst_size_sum.clamp_min(1e-12)

        new_vision = torch.cat((unm_tokens, dst_tokens), dim=1)
        new_sizes = torch.cat((unm_sizes, dst_size_sum), dim=1)

        unm_count = src_tokens.shape[1] - num_merges
        old_to_new = torch.empty(
            batch,
            num_tokens,
            device=vision.device,
            dtype=torch.long,
        )
        dst_new_ids = (
            torch.arange(dst_tokens.shape[1], device=vision.device)
            .unsqueeze(0)
            .expand(batch, -1)
            + unm_count
        )
        old_to_new[:, 1::2] = dst_new_ids
        if plan.unm_idx.numel():
            unm_new_ids = (
                torch.arange(unm_count, device=vision.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            old_to_new.scatter_(1, 2 * plan.unm_idx, unm_new_ids)
        old_to_new.scatter_(
            1,
            2 * plan.src_idx,
            plan.dst_idx + unm_count,
        )
        base_to_current = old_to_new.gather(1, base_to_current)
        return new_vision, new_sizes, base_to_current

    def _global_merge(
        self,
        vision: torch.Tensor,
        sizes: torch.Tensor,
        base_to_current: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        plan = self._global_plan(vision, layer_idx)
        if plan is None:
            return vision, sizes, base_to_current, False
        vision, sizes, base_to_current = self._apply_bipartite_plan(
            vision,
            sizes,
            base_to_current,
            plan,
        )
        return vision, sizes, base_to_current, True

    def _unmerge_to_base(
        self,
        x: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool,
    ) -> torch.Tensor:
        protected = self._num_prefix_tokens + (
            self.num_q if has_queries else 0
        )
        vision = x[:, protected:]
        full_vision = _batched_gather(vision, base_to_current)
        return torch.cat((x[:, :protected], full_vision), dim=1)

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)
        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        sizes, base_to_current = self._initial_merge_state(x)
        selected_layers = set(self.algm_selected_layers)
        query_insert_layer = len(self.encoder.backbone.blocks) - self.num_blocks
        # ToMe variant: force ALL selected layers to the global bipartite cosine merge
        # (merge_started=True skips ALGM's local 2x2 pre-merge). Pure ToMe-style merging.
        merge_started = True
        has_queries = False
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == query_insert_layer:
                x = self._unmerge_to_base(
                    x,
                    base_to_current,
                    has_queries=False,
                )
                x = torch.cat(
                    (
                        self.q.weight[None].expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
                has_queries = True

            attn_mask = None
            if self.masked_attn_enabled and i >= query_insert_layer:
                mask_logits, class_logits = self._predict(
                    self.encoder.backbone.norm(x)
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                attn_mask = self._attn_mask(x, mask_logits, i)

            x = self._run_block_attention(block, x, attn_mask, rope)
            layer_idx = i + 1
            if layer_idx in selected_layers and not has_queries:
                protected = self._num_prefix_tokens
                before = x.shape[1] - protected
                if not merge_started:
                    vision, sizes, base_to_current, changed = self._local_merge(
                        x[:, protected:],
                        sizes,
                        base_to_current,
                    )
                    merge_type = "local"
                    # ALGM switches to global matching after its first selected
                    # layer even when that image batch has no eligible window.
                    merge_started = True
                else:
                    vision, sizes, base_to_current, changed = self._global_merge(
                        x[:, protected:],
                        sizes,
                        base_to_current,
                        layer_idx,
                    )
                    merge_type = "global"
                if changed:
                    x = torch.cat((x[:, :protected], vision), dim=1)
                    if self.merge_logging_enabled and not torch.compiler.is_compiling():
                        warnings.warn(
                            f"[compact-algm:{merge_type}] layer={layer_idx} "
                            f"vision={before}->{vision.shape[1]}"
                        )
            x = self._run_block_mlp(block, x)

        if not has_queries:
            x = self._unmerge_to_base(x, base_to_current, has_queries=False)
            x = torch.cat(
                (self.q.weight[None].expand(x.shape[0], -1, -1), x),
                dim=1,
            )
            has_queries = True

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits_per_layer, class_logits_per_layer
