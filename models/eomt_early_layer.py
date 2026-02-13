# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import torch._dynamo as dynamo

from models.scale_block import ScaleBlock
from algm_addition import apply_patch
from einops import rearrange
from models.algm_layers.global_merge_bavesh import global_bipartite_matching_masked
from models.algm_layers.local_merge import conditional_pooling, merge_source, merge_wavg


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        merge_pool_size: int = 20,
        merged_after_block_idx: int = 20,
        merge_conf_threshold: float = 0.5,
        merge_start_layer: int = 12,
        merge_interval: int = 12,
        query_insert_layer: int = 12,
        query_reinsert_layer: int = 20,
        merge_conf_threshold_end: float = 0.0,
        merge_schedule_epochs: int = 7,
        merge_schedule_enabled: bool = True,
        merge_strategy: str = "project",
        local_merge_enabled: bool = True,
        local_merge_after_layer: int = 1,
        local_merge_window_size: tuple[int, int] = (2, 2),
        local_merge_threshold: float = 0.88,
        merge_logging_enabled: bool = False,
        global_merge_score_metric: str = "js",
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.merge_pool_size = merge_pool_size
        self.merge_after_block_idx = merged_after_block_idx
        self.merge_conf_threshold = merge_conf_threshold
        self.merge_conf_threshold_start = merge_conf_threshold
        self.merge_conf_threshold_end = merge_conf_threshold_end
        self.merge_schedule_epochs = merge_schedule_epochs
        self.merge_schedule_enabled = merge_schedule_enabled
        self.merge_start_layer = merge_start_layer
        self.merge_interval = merge_interval
        self.query_insert_layer = query_insert_layer
        self.query_reinsert_layer = query_reinsert_layer
        self.merge_strategy = merge_strategy
        self.local_merge_enabled = local_merge_enabled
        self.local_merge_after_layer = local_merge_after_layer
        self.local_merge_window_size = local_merge_window_size
        self.local_merge_threshold = local_merge_threshold
        self.merge_logging_enabled = merge_logging_enabled
        self.global_merge_score_metric = global_merge_score_metric
        if self.merge_strategy not in ("project", "rematch"):
            raise ValueError("merge_strategy must be one of: 'project', 'rematch'")
        if self.global_merge_score_metric not in ("cosine", "js", "hellinger", "dot"):
            raise ValueError(
                "global_merge_score_metric must be one of: "
                "'cosine', 'js', 'hellinger', 'dot'"
            )

        self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

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
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

        self.selected_layers = [1, 7]
        # self._algm_init()

    def _should_merge(self, layer_idx: int) -> bool:
        if self.merge_interval <= 0:
            return False
        if layer_idx < self.merge_start_layer:
            return False
        return (layer_idx - self.merge_start_layer) % self.merge_interval == 0

    def update_merge_schedule(self, epoch: int, max_epochs: Optional[int] = None) -> None:
        if not self.merge_schedule_enabled:
            return
        total_epochs = self.merge_schedule_epochs or max_epochs or 1
        total_epochs = max(int(total_epochs), 1)
        if total_epochs == 1:
            t = 1.0
        else:
            t = min(max(int(epoch), 0), total_epochs - 1) / float(total_epochs - 1)
        start = float(self.merge_conf_threshold_start)
        end = float(self.merge_conf_threshold_end)
        self.merge_conf_threshold = start + t * (end - start)

    def _algm_init(self):
        apply_patch(
            self.encoder,
            selected_layers=self.selected_layers,
            trace_source=True,
            prop_attn=True,
        )

        # assert False, f'{self.encoder.backbone._turbo_info}'

    def _reset_turbo_info(self):
        """Reset ALGM runtime state each forward."""
        backbone = getattr(self.encoder, "backbone", None)
        turbo_info = getattr(backbone, "_turbo_info", None)
        if turbo_info is None:
            raise ValueError("ALGM patch not applied to backbone.")

        turbo_info["size"] = None
        turbo_info["source"] = None
        turbo_info['rel_pos'] = None
        turbo_info["selected_layers"] = list(self.selected_layers)
        turbo_info["window_size"] = getattr(backbone, "window_size", None)
        turbo_info["threshold"] = getattr(backbone, "threshold", None)

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        # assert False, f'SAHPEEE! , {x.shape}, {self.encoder.backbone.num_prefix_tokens}'
        # print(f'SAHPEEE! , {x.shape}, {self.encoder.backbone.num_prefix_tokens}')
        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        # print('SAHPEEE LASTT!', x.shape)
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def _merge_by_info(self, x: torch.Tensor, merge_info: dict, mode: str = "mean") -> torch.Tensor:
        if merge_info.get("type") != "bipartite":
            raise ValueError("Unknown merge info for merge projection.")
        unm_idx = merge_info["unm_idx"]
        src_idx = merge_info["src_idx"]
        dst_idx = merge_info["dst_idx"]

        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        r = src_idx.shape[1]
        if r == 0:
            return x
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def _project_mask_probs(self, mask_probs: torch.Tensor, merge_history: list) -> torch.Tensor:
        if not merge_history:
            return mask_probs
        probs = mask_probs.transpose(1, 2)  # [B, N, Q]
        for merge_info in merge_history:
            probs = self._merge_by_info(probs, merge_info, mode="mean")
        return probs.transpose(1, 2)

    def _project_mask_probs_local(
        self,
        mask_probs: torch.Tensor,
        local_merge_source: torch.Tensor,
        has_queries: bool = True,
    ) -> torch.Tensor:
        if local_merge_source is None:
            return mask_probs
        protected = (self.num_q if has_queries else 0) + self.encoder.backbone.num_prefix_tokens
        source_vis = local_merge_source[:, protected:, protected:]  # [B, M, N]
        if source_vis.numel() == 0:
            return mask_probs
        s = source_vis.float()
        denom = s.sum(dim=2, keepdim=False).clamp(min=1.0)  # [B, M]
        # mask_probs: [B, Q, N] -> [B, Q, M]
        merged = torch.einsum("bqn,bmn->bqm", mask_probs, s)
        merged = merged / denom.unsqueeze(1)
        return merged

    def _merge_tokens(
        self,
        x: torch.Tensor,
        layer_idx: int,
        merge_history: Optional[list] = None,
        merge_strategy: str = "project",
        local_merge_source: Optional[torch.Tensor] = None,
    ):
        # Compute token merge masks. If we already merged, unmerge for mask prediction
        # but keep using the current merged tokens for the new merge.
        x_for_merge = x
        if merge_history or local_merge_source is not None:
            x_unmerged = x
            if merge_history:
                for merged_mask in reversed(merge_history):
                    x_unmerged = self._unmerge_tokens(x_unmerged, merged_mask)
            if local_merge_source is not None:
                x_unmerged = self._algm_unmerge(x_unmerged, local_merge_source, has_queries=True)
            mask_logits, _ = self._predict(self.encoder.backbone.norm(x_unmerged)) # [B, N, 128, 128]
            if merge_strategy == "rematch":
                x_for_merge = x_unmerged
        else:
            mask_logits, _ = self._predict(self.encoder.backbone.norm(x)) # [B, N, 128, 128]

        grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
        interpolated = F.interpolate(
            mask_logits,
            (grid_h, grid_w),
            mode="bilinear",
        ) # [B, N, 32, 32]
        vision_token_masks = torch.softmax(interpolated, dim=1).flatten(2)
        if merge_history and merge_strategy == "project":
            vision_token_masks = self._project_mask_probs(vision_token_masks, merge_history)
            protected = self.num_q + self.encoder.backbone.num_prefix_tokens
            vision_len = x.shape[1] - protected
            if not dynamo.is_compiling():
                if vision_token_masks.shape[-1] != vision_len:
                    warnings.warn(
                        f"Projected mask length {vision_token_masks.shape[-1]} "
                        f"!= current vision length {vision_len} at layer {layer_idx}."
                    )
                assert vision_token_masks.shape[-1] == vision_len, (
                    f"Mask projection mismatch at layer {layer_idx}: "
                    f"{vision_token_masks.shape[-1]} vs {vision_len}"
                )

        protected = self.num_q + self.encoder.backbone.num_prefix_tokens
        vision_tokens = x_for_merge[:, protected:, :]
        if local_merge_source is not None and vision_token_masks.shape[-1] != vision_tokens.shape[1]:
            vision_token_masks = self._project_mask_probs_local(
                vision_token_masks, local_merge_source, has_queries=True
            )
            if not dynamo.is_compiling():
                if vision_token_masks.shape[-1] != vision_tokens.shape[1]:
                    warnings.warn(
                        f"Local mask projection mismatch {vision_token_masks.shape[-1]} "
                        f"!= merged vision length {vision_tokens.shape[1]} at layer {layer_idx}."
                    )
                assert vision_token_masks.shape[-1] == vision_tokens.shape[1], (
                    f"Local mask projection mismatch at layer {layer_idx}: "
                    f"{vision_token_masks.shape[-1]} vs {vision_tokens.shape[1]}"
                )
        merge, merge_info = global_bipartite_matching_masked(
            vision_tokens,
            vision_token_masks,
            layer_idx=layer_idx,
            tau=self.merge_conf_threshold,
            score_metric=self.global_merge_score_metric,
        )
        merged_vision = merge(vision_tokens)
        x = torch.cat((x_for_merge[:, :protected, :], merged_vision), dim=1)
        if self.merge_logging_enabled and not dynamo.is_compiling():
            warnings.warn(
                f"[global-merge] layer={layer_idx} vision_tokens={vision_tokens.shape[1]} "
                f"merged_vision={merged_vision.shape[1]} total_tokens={x.shape[1]}"
            )

        if merge_info is None:
            return x, None

        merge_info["type"] = "bipartite"
        return x, merge_info

    def _unmerge_tokens(self, x: torch.Tensor, mask):
        # unmerge tokens
        if isinstance(mask, dict):
            if mask.get("type") != "bipartite":
                raise ValueError("Unknown merge info for unmerge.")
            unm_idx = mask["unm_idx"]
            src_idx = mask["src_idx"]
            dst_idx = mask["dst_idx"]
            num_patch = mask["num_patch"]

            protected = self.num_q + self.encoder.backbone.num_prefix_tokens
            merged = x[:, protected:, :]
            if merged.shape[1] == num_patch:
                return x

            unm_len = unm_idx.shape[1]
            dst = merged[:, unm_len:, :]
            unm = merged[:, :unm_len, :]
            n, _, c = merged.shape
            r = src_idx.shape[1]

            src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
            out_patch = torch.zeros(n, num_patch, c, device=x.device, dtype=x.dtype)
            out_patch[..., 1::2, :] = dst
            out_patch.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            out_patch.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
            reprojected_tokens = out_patch
        else:
            fused = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
            fused_expanded = fused.unsqueeze(2)
            mask_expanded = mask.unsqueeze(-1)
            reproj_sum = (mask_expanded * fused_expanded).sum(dim=1)
            reproj_count = mask.sum(dim=1, keepdim=False).clamp(min=1e-6).unsqueeze(-1)  # [B, HW, 1]
            reprojected_tokens = reproj_sum / reproj_count

        x = torch.cat(
            (x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], reprojected_tokens), dim=1
        )
        return x
    
    def _algm_unmerge(self, x, source, has_queries=True):
        if has_queries:
            protected = self.num_q + self.encoder.backbone.num_prefix_tokens
        else:
            protected = self.encoder.backbone.num_prefix_tokens

        # Rows = merged tokens; Cols = original tokens (before merges). Align rows with current tokens.
        if source.shape[1] != x.shape[1]:
            raise ValueError("Source and x have different number of tokens.")

        src_vis = source[:, protected:, protected:]  # rows: merged vision, cols: original vision
        vision = x[:, protected:, :]

        if src_vis.shape[1] > vision.shape[1]:
            raise ValueError("Source has more merged tokens than x has vision tokens.")

        idxs = src_vis.argmax(dim=1)  # [B, original_vision_len]

        target_vision_len = (
            self.encoder.backbone.patch_embed.grid_size[0]
            * self.encoder.backbone.patch_embed.grid_size[1]
        )
        if idxs.shape[1] != target_vision_len:
            raise ValueError("Number of original vision tokens does not match target.")

        x_ = torch.gather(
            vision,
            1,
            idxs.unsqueeze(-1).expand(-1, idxs.shape[1], vision.shape[2]),
        )

        x = torch.cat((x[:, :protected, :], x_), dim=1)

        return x

    # def _algm_unmerge(self, x: torch.Tensor, source: torch.Tensor, has_queries: bool = True):
    #     GS = self.encoder.backbone.patch_embed.grid_size[0]
    #     # x = self.head(x)
    #     if source is not None :
            
    #         idxs = source[:, self.encoder.backbone.num_prefix_tokens:, self.encoder.backbone.num_prefix_tokens:].argmax(dim=1)
    #         # defining a torch tensor with the output size 
    #         masks_shape = x.shape
    #         size_x = source.shape[2]-self.encoder.backbone.num_prefix_tokens
    #         x_ =  torch.ones(masks_shape[0],size_x,masks_shape[2],device=x.device)
        
    #         print('LASTTT!!',x_.shape, x.shape, idxs.shape)
    #         assert False, f'LASTTT!! {x_.shape}, {x.shape}, {idxs.shape}'
    #         for batch in range(0,masks_shape[0]):
    #             x_[batch,:,:] = x[batch,idxs[batch]]

    #             # x_ = torch.cat((x[:, :self.encoder.backbone.num_prefix_tokens, :], x_), dim=1)
    #             # x_ = torch.cat((source[:]))
            
    #         x_ = torch.cat((x[:, :self.encoder.backbone.num_prefix_tokens, :], x_), dim=1)
    #     else:
    #             x_ = x

    #     # x_ = rearrange(x_, "b (h w) c -> b c h w", h=GS)
    #     return x_

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # self._reset_turbo_info()
        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        mask_logits_per_layer, class_logits_per_layer = [], []
        turbo_info = getattr(self.encoder.backbone, "_turbo_info", None)
        num_protected_tokens = self.encoder.backbone.num_prefix_tokens
        source = turbo_info["source"] if turbo_info is not None else None
        merged_masks = []
        local_merge_source = None
        local_merge_size = None
        local_merged = False
        # merge_after_block_idx = 20  # 0-based index; adjust if you want a different layer.

        has_queries = False
        queries_persistent = False
        if self.query_insert_layer is None:
            query_insert_layer = len(self.encoder.backbone.blocks) - self.num_blocks
        else:
            query_insert_layer = max(int(self.query_insert_layer) - 1, 0)
            query_insert_layer = min(query_insert_layer, len(self.encoder.backbone.blocks) - 1)
        if self.query_reinsert_layer is None:
            query_reinsert_layer = len(self.encoder.backbone.blocks) - self.num_blocks
        else:
            query_reinsert_layer = max(int(self.query_reinsert_layer) - 1, 0)
            query_reinsert_layer = min(query_reinsert_layer, len(self.encoder.backbone.blocks) - 1)

        def _prepend_queries(
            x_tokens: torch.Tensor,
            source_map: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if local_merged and source_map is not None and self.num_q > 0:
                if source_map.shape[1] != x_tokens.shape[1] + self.num_q:
                    padded_source = source_map.new_zeros(
                        source_map.shape[0],
                        source_map.shape[1] + self.num_q,
                        source_map.shape[2] + self.num_q,
                    )
                    padded_source[:, self.num_q :, self.num_q :] = source_map
                    diag_idx = torch.arange(self.num_q, device=source_map.device)
                    padded_source[:, diag_idx, diag_idx] = 1
                    source_map = padded_source
            x_tokens = torch.cat(
                (self.q.weight[None, :, :].expand(x_tokens.shape[0], -1, -1), x_tokens), dim=1
            )
            return x_tokens, source_map

        for i, block in enumerate(self.encoder.backbone.blocks):
            # if i>=8:
            #     assert False, 'Local merge should be done after 1'
            if i == query_insert_layer and not has_queries:
                x, local_merge_source = _prepend_queries(x, local_merge_source)
                has_queries = True
                queries_persistent = i == query_reinsert_layer
            elif i == query_reinsert_layer and not has_queries:
                x, local_merge_source = _prepend_queries(x, local_merge_source)
                has_queries = True
                queries_persistent = True
                # if self.encoder.backbone._turbo_info["source"] is not None:
                    # x = self._algm_unmerge(x, self.encoder.backbone._turbo_info["source"], has_queries=False)
                    # assert False
                # num_protected_tokens += self.num_q
                # target_tokens = x.shape[1] + self.num_q
                # if source is not None and source.shape[1] != target_tokens:
                #     # Expand source to include queries: rows and cols grow by num_q
                #     padded_source = source.new_zeros(
                #         source.shape[0],
                #         source.shape[1] + self.num_q,
                #         source.shape[2] + self.num_q,
                #     )
                #     padded_source[:, self.num_q :, self.num_q :] = source
                #     if self.num_q > 0:
                #         diag_idx = torch.arange(self.num_q, device=source.device)
                #         padded_source[:, diag_idx, diag_idx] = 1
                #     turbo_info["source"] = padded_source
                #     source = padded_source
                # print('SAHPEEE!', x.shape)

            if hasattr(block, "_turbo_info"):
                # print('i:', i)
                x = block(x, num_protected_tokens=num_protected_tokens)
                # source = self.encoder.backbone._turbo_info["source"]
            else:
                x = block(x)
            if (
                self.local_merge_enabled
                and not has_queries
                and not local_merged
                and (i + 1) == max(int(self.local_merge_after_layer), 1)
            ):
                before_tokens = x.shape[1]
                merge = conditional_pooling(
                    x,
                    self.local_merge_threshold,
                    self.local_merge_window_size,
                    num_prefix_tokens=num_protected_tokens,
                )
                local_merge_source = merge_source(merge, x, local_merge_source)
                x, local_merge_size = merge_wavg(merge, x, local_merge_size)
                local_merged = True
                if self.merge_logging_enabled and not dynamo.is_compiling():
                    warnings.warn(
                        f"[local-merge] layer={i + 1} tokens={before_tokens}->{x.shape[1]}"
                    )
            if has_queries and i < len(self.encoder.backbone.blocks) - 1 and self._should_merge(i + 1):
                x, merged_mask = self._merge_tokens(
                    x,
                    layer_idx=i + 1,
                    merge_history=merged_masks,
                    merge_strategy=self.merge_strategy,
                    local_merge_source=local_merge_source,
                )
                if self.merge_strategy == "rematch" and local_merge_source is not None:
                    local_merged = False
                    local_merge_source = None
                    local_merge_size = None
                if merged_mask is not None:
                    if self.merge_strategy == "rematch":
                        merged_masks = [merged_mask]
                    else:
                        merged_masks.append(merged_mask)
                elif self.merge_strategy == "rematch":
                    merged_masks = []
            if has_queries and not queries_persistent and i == query_insert_layer:
                x = x[:, self.num_q :, :]
                has_queries = False
            # if i == len(self.encoder.backbone.blocks) - self.num_blocks:
            #     x, merged_mask = self._merge_tokens(x, layer_idx=i + 1)
        # source = turbo_info["source"]
        #### This is code for ALGM token merging baseline. ######
        # if self.encoder.backbone._turbo_info["source"] is not None:
        #     x_vis = x[:, self.num_q:, :]
        #     x_vis = self._algm_unmerge(x_vis, self.encoder.backbone._turbo_info["source"], has_queries=False)
        #     x = torch.cat((x[:, :self.num_q, :], x_vis), dim=1)
        #### End of ALGM token merging baseline. ######

        if merged_masks or local_merged:
            for merged_mask in reversed(merged_masks):
                x = self._unmerge_tokens(x, merged_mask)
            if local_merged and local_merge_source is not None:
                x = self._algm_unmerge(x, local_merge_source, has_queries=has_queries)
            if not dynamo.is_compiling():
                expected_vision_len = (
                    self.encoder.backbone.patch_embed.grid_size[0]
                    * self.encoder.backbone.patch_embed.grid_size[1]
                )
                protected = (self.num_q if has_queries else 0) + self.encoder.backbone.num_prefix_tokens
                actual_vision_len = x.shape[1] - protected
                if self.merge_logging_enabled:
                    warnings.warn(
                        f"[unmerge] vision_tokens={actual_vision_len} expected={expected_vision_len} "
                        f"protected={protected} total_tokens={x.shape[1]}"
                    )
                if actual_vision_len != expected_vision_len:
                    warnings.warn(
                        f"Unmerge produced {actual_vision_len} vision tokens, "
                        f"expected {expected_vision_len}."
                    )
                assert actual_vision_len == expected_vision_len, (
                    f"Unmerge vision token mismatch: {actual_vision_len} vs {expected_vision_len}"
                )

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
