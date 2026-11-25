# ---------------------------------------------------------------
# Original-ALGM-style EoMT panoptic adapter.
#
# This keeps the learnable EoMT parameter structure identical to models.eomt_ori
# so existing panoptic checkpoints load unchanged. The merge logic is adapted
# from third_party/algm-segmenter/algm/algm:
# local conditional pooling at the first selected layer, then feature-only
# adaptive bipartite matching at later selected layers.
# ---------------------------------------------------------------

from typing import Optional
import warnings

import torch
import torch.nn as nn

from models.eomt_ori import EoMT as BaseEoMT
from models.algm_layers.local_merge import conditional_pooling, merge_source, merge_wavg


def _do_nothing(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
    del mode
    return x


def _original_algm_global_matching(
    metric: torch.Tensor,
    layer_idx: int,
    num_prefix_tokens: int,
):
    """Feature-only ALGM global matching, with EoMT prefix tokens protected."""
    protected = int(num_prefix_tokens)
    vision_metric = metric[:, protected:, :]
    num_tokens = vision_metric.shape[1]
    if num_tokens <= 1:
        return _do_nothing

    with torch.no_grad():
        metric_norm = vision_metric / (vision_metric.norm(dim=-1, keepdim=True) + 1e-12)
        src_metric = metric_norm[..., ::2, :]
        dst_metric = metric_norm[..., 1::2, :]
        if src_metric.shape[1] == 0 or dst_metric.shape[1] == 0:
            return _do_nothing

        scores = src_metric @ dst_metric.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Original ALGM uses a batch-level adaptive threshold:
        # mean(score) + std(score) / layer_idx.
        denom = max(int(layer_idx), 1)
        threshold_scores = node_max[:, 1:] if node_max.shape[1] > 1 else node_max
        node_threshold = threshold_scores.mean(dim=1).mean()
        node_threshold = node_threshold + threshold_scores.std(dim=1, unbiased=False).mean() / denom
        r = torch.ge(node_max, node_threshold).sum(dim=1).min()
        r = int(r.item())
        if r <= 0:
            return _do_nothing

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        prefix_tokens = x[:, :protected, :]
        vision = x[:, protected:, :]
        src, dst = vision[..., ::2, :], vision[..., 1::2, :]
        batch, src_len, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(batch, src_len - r, channels))
        src = src.gather(dim=-2, index=src_idx.expand(batch, r, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(batch, r, channels), src, reduce=mode)
        return torch.cat((prefix_tokens, unm, dst), dim=1)

    return merge


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
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
        )
        if algm_selected_layers is None:
            algm_selected_layers = [1, 7]
        self.algm_selected_layers = [int(layer) for layer in algm_selected_layers]
        self.local_merge_threshold = float(local_merge_threshold)
        self.local_merge_window_size = tuple(local_merge_window_size)
        self.merge_logging_enabled = bool(merge_logging_enabled)

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

    def _algm_merge(
        self,
        x: torch.Tensor,
        source: Optional[torch.Tensor],
        size: Optional[torch.Tensor],
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        before = x.shape[1] - self._num_prefix_tokens
        if source is None:
            merge = conditional_pooling(
                x,
                self.local_merge_threshold,
                self.local_merge_window_size,
                num_prefix_tokens=self._num_prefix_tokens,
                grid_hw=self._grid_size,
            )
            merge_type = "local"
        else:
            merge = _original_algm_global_matching(
                x,
                layer_idx=layer_idx,
                num_prefix_tokens=self._num_prefix_tokens,
            )
            merge_type = "global"

        source = merge_source(merge, x, source)
        x, size = merge_wavg(merge, x, size)

        if self.merge_logging_enabled and not torch.compiler.is_compiling():
            after = x.shape[1] - self._num_prefix_tokens
            warnings.warn(f"[original-algm:{merge_type}] layer={layer_idx} vision={before}->{after}")
        return x, source, size

    def _unmerge_with_source(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        has_queries: bool,
    ) -> torch.Tensor:
        source_protected = self._num_prefix_tokens
        x_protected = self._num_prefix_tokens + (self.num_q if has_queries else 0)
        source_vis = source[:, source_protected:, source_protected:]
        vision = x[:, x_protected:, :]
        if source_vis.shape[0] != vision.shape[0]:
            raise ValueError("Source and vision batch sizes differ.")
        if source_vis.shape[1] != vision.shape[1]:
            raise ValueError(
                "Source rows must match current merged vision tokens: "
                f"{source_vis.shape[1]} vs {vision.shape[1]}"
            )

        idxs = source_vis.argmax(dim=1)
        full_vision = torch.gather(
            vision,
            1,
            idxs.unsqueeze(-1).expand(-1, idxs.shape[1], vision.shape[2]),
        )
        return torch.cat((x[:, :x_protected, :], full_vision), dim=1)

    def _maybe_unmerge_before_queries(
        self,
        x: torch.Tensor,
        source: Optional[torch.Tensor],
        size: Optional[torch.Tensor],
        has_queries: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if source is None:
            return x, source, size
        x = self._unmerge_with_source(x, source, has_queries=has_queries)
        if self.merge_logging_enabled and not torch.compiler.is_compiling():
            warnings.warn("[original-algm:unmerge] restored full patch grid before EoMT queries")
        return x, None, None

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        ps = self.encoder.backbone.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else ps
        target_grid = (x.shape[-2] // ps, x.shape[-1] // ps)
        if not hasattr(self, "_creation_grid_size"):
            self._creation_grid_size = tuple(
                self.encoder.backbone.patch_embed.grid_size
            )

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        # _pos_embed uses grid_size as the OLD size for pos-embed interpolation,
        # so restore the creation grid here (a prior forward may have left it at
        # that call's target). After, set the ACTUAL input grid, which _grid_size
        # (every merge/readout step below) reads. Enables eval at a resolution
        # other than the creation one.
        self.encoder.backbone.patch_embed.grid_size = self._creation_grid_size
        x = self.encoder.backbone.patch_embed(x)
        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)
        self.encoder.backbone.patch_embed.grid_size = target_grid

        mask_logits_per_layer, class_logits_per_layer = [], []
        query_insert_layer = len(self.encoder.backbone.blocks) - self.num_blocks
        selected_layers = set(self.algm_selected_layers)
        source = None
        size = None
        has_queries = False
        attn_mask = None

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == query_insert_layer:
                x, source, size = self._maybe_unmerge_before_queries(
                    x,
                    source,
                    size,
                    has_queries=False,
                )
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x),
                    dim=1,
                )
                has_queries = True

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                attn_mask = self._attn_mask(x, mask_logits, i)
            else:
                attn_mask = None

            layer_idx = i + 1
            x = self._run_block_attention(block, x, attn_mask, rope=rope)
            if layer_idx in selected_layers and not has_queries:
                x, source, size = self._algm_merge(x, source, size, layer_idx)
            elif layer_idx in selected_layers and has_queries and not torch.compiler.is_compiling():
                warnings.warn(
                    f"Skipping original ALGM layer {layer_idx}; this adapter only merges before EoMT queries."
                )
            x = self._run_block_mlp(block, x)

        x, source, size = self._maybe_unmerge_before_queries(x, source, size, has_queries=has_queries)
        del source, size

        expected_vision = self._grid_size[0] * self._grid_size[1]
        actual_vision = x.shape[1] - self._num_prefix_tokens - (self.num_q if has_queries else 0)
        if not torch.compiler.is_compiling() and actual_vision != expected_vision:
            raise RuntimeError(
                f"Original ALGM unmerge produced {actual_vision} vision tokens, "
                f"expected {expected_vision}."
            )

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer
