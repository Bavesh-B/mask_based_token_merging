# ---------------------------------------------------------------
# Merged-tail ToMe (pure global bipartite cosine merging) for EoMT.
#
# Same architecture as eomt_merged_tail_algm (keeps the merged token set through
# the decoder / query blocks, unmerge only before each readout -- the design that
# makes MaskMerge fast), but:
#   * the local 2x2 pre-merge is disabled: every selected layer runs the GLOBAL
#     bipartite cosine merge (pure ToMe, no locality prior, no mask gate);
#   * a global_merge_ratio knob merges a FIXED fraction of src tokens per layer
#     (canonical ToMe), and global_threshold_scale scales the adaptive threshold,
#     so ToMe can be pushed to ANY budget -> a full accuracy-FLOPs curve down to
#     MaskMerge's operating point.
# This is the fair cosine baseline for MaskMerge: identical merged-tail compute,
# differs only in the merge criterion (cosine similarity vs mask-constrained).
# ---------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch

from models.eomt_compact_algm import BipartitePlan
from models.eomt_merged_tail_algm import EoMT as MergedTailAlgmEoMT


class EoMT(MergedTailAlgmEoMT):
    def __init__(self, *args, global_merge_ratio: float = 0.0,
                 global_threshold_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_merge_ratio = float(global_merge_ratio)
        self.global_threshold_scale = float(global_threshold_scale)

    def _local_merge(self, vision, sizes, base_to_current):
        # ToMe has no local 2x2 pre-merge: run the global bipartite cosine merge
        # at the first selected layer instead, so all merges are pure cosine.
        return self._global_merge(vision, sizes, base_to_current,
                                  self.algm_selected_layers[0])

    def _global_plan(self, vision: torch.Tensor, layer_idx: int) -> Optional[BipartitePlan]:
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
            threshold_scores = node_max[:, 1:] if node_max.shape[1] > 1 else node_max
            threshold = threshold_scores.mean(dim=1).mean()
            threshold = threshold + threshold_scores.std(dim=1, unbiased=False).mean() / max(int(layer_idx), 1)
            threshold = threshold * self.global_threshold_scale
            if self.global_merge_ratio > 0.0:
                # canonical ToMe: merge a fixed fraction of the src tokens (the
                # top-r most-similar bipartite pairs), regardless of threshold.
                num_src = node_max.shape[1]
                num_merges = int(min(num_src, max(0, round(self.global_merge_ratio * num_src))))
            else:
                num_merges = int((node_max >= threshold).sum(dim=1).min().item())
            if num_merges <= 0:
                return None
            edge_idx = node_max.argsort(dim=-1, descending=True)
            src_idx = edge_idx[:, :num_merges]
            unm_idx = edge_idx[:, num_merges:]
            dst_idx = node_idx.gather(1, src_idx)
            return BipartitePlan(src_idx, dst_idx, unm_idx)
