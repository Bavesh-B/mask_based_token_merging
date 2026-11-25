"""ALGM that keeps merged tokens through the query-refinement blocks.

Diagnosis this responds to
--------------------------
Per-block token counts at batch 1 (ADE20K val, identical images):

    block        dense    ALGM    MaskMerge
    0             1605    1605         1605
    1-5           1605    1236         1166
    7-18          1605     931          918
    20-23         1805    1805         1118      <- the difference
    sum          39320   28347        25402
    % of dense    100%   72.1%        64.6%

`eomt_compact_algm.forward` calls `_unmerge_to_base` at the query-insert layer,
so the last four blocks run at FULL resolution: 4 x 1805 = 7220 token-blocks,
25.5% of ALGM's entire compute, at roughly twice the density of the merged
region. MaskMerge does not do this, and that single choice -- not its merge
criterion -- is where its ~10% compute advantage over ALGM comes from.

The unmerge is only needed because `_predict` reshapes patch tokens into the
patch grid (`x.transpose(1, 2).reshape(B, C, grid_h, grid_w)`), which requires
one token per grid cell. That is a requirement of the READOUT, not of the
transformer blocks.

Mechanism
---------
Insert the queries without unmerging, run the refinement blocks on the merged
token set, and unmerge only immediately before each `_predict` call. The gather
that unmerges is O(N) and negligible next to four transformer blocks.

Merge decisions, merge layers, thresholds and weights are all identical to
`eomt_compact_algm`, so a difference in accuracy is attributable to running the
refinement blocks at merged resolution and nothing else.

Expected: ~9.5% fewer token-blocks, and more than that in wall-clock, since
attention is superlinear in sequence length.

Note on masked attention: this variant leaves `masked_attn_enabled` alone but the
per-layer attention mask is built from the unmerged grid, so it costs an extra
unmerge per refinement block when enabled. At inference this is moot -- released
checkpoints store `attn_mask_probs = [0, 0, 0, 0]`, and `_disable_attn_mask`
with prob 0 unmasks everything -- so evaluate with `masked_attn_enabled: False`
for the cheapest equivalent path.
"""

from __future__ import annotations

import warnings

import torch

from models.eomt_compact_algm import EoMT as CompactAlgmEoMT


class EoMT(CompactAlgmEoMT):
    def _predict_unmerged(
        self,
        x: torch.Tensor,
        base_to_current: torch.Tensor,
    ):
        """Unmerge patch tokens to the full grid, then run the normal readout."""
        full = self._unmerge_to_base(x, base_to_current, has_queries=True)
        return self._predict(self.encoder.backbone.norm(full))

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
        merge_started = False
        has_queries = False
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == query_insert_layer:
                # The parent unmerges to the full grid here. We do not: the
                # queries are concatenated to the MERGED token set.
                x = torch.cat(
                    (self.q.weight[None].expand(x.shape[0], -1, -1), x),
                    dim=1,
                )
                has_queries = True

            attn_mask = None
            if self.masked_attn_enabled and i >= query_insert_layer:
                mask_logits, class_logits = self._predict_unmerged(
                    x, base_to_current
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                # `_attn_mask` indexes patch tokens positionally against the full
                # grid, so it needs the unmerged sequence too.
                attn_mask = self._attn_mask(
                    self._unmerge_to_base(x, base_to_current, has_queries=True),
                    mask_logits,
                    i,
                )

            x = self._run_block_attention(block, x, attn_mask, rope)
            layer_idx = i + 1
            if layer_idx in selected_layers and not has_queries:
                protected = self._num_prefix_tokens
                before = x.shape[1] - protected
                if not merge_started:
                    vision, sizes, base_to_current, changed = self._local_merge(
                        x[:, protected:], sizes, base_to_current
                    )
                    merge_type = "local"
                    merge_started = True
                else:
                    vision, sizes, base_to_current, changed = self._global_merge(
                        x[:, protected:], sizes, base_to_current, layer_idx
                    )
                    merge_type = "global"
                if changed:
                    x = torch.cat((x[:, :protected], vision), dim=1)
                    if (
                        self.merge_logging_enabled
                        and not torch.compiler.is_compiling()
                    ):
                        warnings.warn(
                            f"[merged-tail-algm:{merge_type}] layer={layer_idx} "
                            f"vision={before}->{vision.shape[1]}"
                        )
            x = self._run_block_mlp(block, x)

        if not has_queries:
            x = torch.cat(
                (self.q.weight[None].expand(x.shape[0], -1, -1), x), dim=1
            )
            has_queries = True

        mask_logits, class_logits = self._predict_unmerged(x, base_to_current)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits_per_layer, class_logits_per_layer
