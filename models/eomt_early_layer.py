# # ---------------------------------------------------------------
# # © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# # Licensed under the MIT License.
# #
# # Portions of this file are adapted from the timm library by Ross Wightman,
# # used under the Apache 2.0 License.
# # ---------------------------------------------------------------

# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import warnings
# import torch._dynamo as dynamo

# from models.scale_block import ScaleBlock
# from algm_addition import apply_patch
# from einops import rearrange
# from models.algm_layers.global_merge_bavesh import global_bipartite_matching_masked
# from models.algm_layers.local_merge import conditional_pooling, merge_source, merge_wavg


# class EoMT(nn.Module):
#     def __init__(
#         self,
#         encoder: nn.Module,
#         num_classes,
#         num_q,
#         num_blocks=4,
#         masked_attn_enabled=True,
#         merge_pool_size: int = 20,
#         merged_after_block_idx: int = 20,
#         merge_conf_threshold: float = 0.5,
#         merge_start_layer: int = 12,
#         merge_interval: int = 12,
#         query_insert_layer: int = 12,
#         query_reinsert_layer: int = 20,
#         merge_conf_threshold_end: float = 0.0,
#         merge_schedule_epochs: int = 7,
#         merge_schedule_enabled: bool = True,
#         merge_strategy: str = "project",
#         local_merge_enabled: bool = True,
#         local_merge_after_layer: int = 1,
#         local_merge_window_size: tuple[int, int] = (2, 2),
#         local_merge_threshold: float = 0.88,
#         merge_logging_enabled: bool = False,
#         global_merge_score_metric: str = "dot",
#     ):
#         super().__init__()
#         self.encoder = encoder
#         self.num_q = num_q
#         self.num_blocks = num_blocks
#         self.masked_attn_enabled = masked_attn_enabled
#         self.merge_pool_size = merge_pool_size
#         self.merge_after_block_idx = merged_after_block_idx
#         self.merge_conf_threshold = merge_conf_threshold
#         self.merge_conf_threshold_start = merge_conf_threshold
#         self.merge_conf_threshold_end = merge_conf_threshold_end
#         self.merge_schedule_epochs = merge_schedule_epochs
#         self.merge_schedule_enabled = merge_schedule_enabled
#         self.merge_start_layer = merge_start_layer
#         self.merge_interval = merge_interval
#         self.query_insert_layer = query_insert_layer
#         self.query_reinsert_layer = query_reinsert_layer
#         self.merge_strategy = merge_strategy
#         self.local_merge_enabled = local_merge_enabled
#         self.local_merge_after_layer = local_merge_after_layer
#         self.local_merge_window_size = local_merge_window_size
#         self.local_merge_threshold = local_merge_threshold
#         self.merge_logging_enabled = merge_logging_enabled
#         self.global_merge_score_metric = global_merge_score_metric
#         if self.merge_strategy not in ("project", "rematch"):
#             raise ValueError("merge_strategy must be one of: 'project', 'rematch'")
#         if self.global_merge_score_metric not in ("cosine", "js", "hellinger", "dot"):
#             raise ValueError(
#                 "global_merge_score_metric must be one of: "
#                 "'cosine', 'js', 'hellinger', 'dot'"
#             )

#         self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))

#         self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

#         self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

#         self.mask_head = nn.Sequential(
#             nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
#         )

#         patch_size = encoder.backbone.patch_embed.patch_size
#         max_patch_size = max(patch_size[0], patch_size[1])
#         num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

#         self.upscale = nn.Sequential(
#             *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
#         )

#         self.selected_layers = [1, 7]
#         # self._algm_init()

#     def _should_merge(self, layer_idx: int) -> bool:
#         if self.merge_interval <= 0:
#             return False
#         if layer_idx < self.merge_start_layer:
#             return False
#         return (layer_idx - self.merge_start_layer) % self.merge_interval == 0

#     def update_merge_schedule(self, epoch: int, max_epochs: Optional[int] = None) -> None:
#         if not self.merge_schedule_enabled:
#             return
#         total_epochs = self.merge_schedule_epochs or max_epochs or 1
#         total_epochs = max(int(total_epochs), 1)
#         if total_epochs == 1:
#             t = 1.0
#         else:
#             t = min(max(int(epoch), 0), total_epochs - 1) / float(total_epochs - 1)
#         start = float(self.merge_conf_threshold_start)
#         end = float(self.merge_conf_threshold_end)
#         self.merge_conf_threshold = start + t * (end - start)

#     def _algm_init(self):
#         apply_patch(
#             self.encoder,
#             selected_layers=self.selected_layers,
#             trace_source=True,
#             prop_attn=True,
#         )

#         # assert False, f'{self.encoder.backbone._turbo_info}'

#     def _reset_turbo_info(self):
#         """Reset ALGM runtime state each forward."""
#         backbone = getattr(self.encoder, "backbone", None)
#         turbo_info = getattr(backbone, "_turbo_info", None)
#         if turbo_info is None:
#             raise ValueError("ALGM patch not applied to backbone.")

#         turbo_info["size"] = None
#         turbo_info["source"] = None
#         turbo_info['rel_pos'] = None
#         turbo_info["selected_layers"] = list(self.selected_layers)
#         turbo_info["window_size"] = getattr(backbone, "window_size", None)
#         turbo_info["threshold"] = getattr(backbone, "threshold", None)

#     def _predict(self, x: torch.Tensor):
#         q = x[:, : self.num_q, :]

#         class_logits = self.class_head(q)

#         # assert False, f'SAHPEEE! , {x.shape}, {self.encoder.backbone.num_prefix_tokens}'
#         # print(f'SAHPEEE! , {x.shape}, {self.encoder.backbone.num_prefix_tokens}')
#         x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
#         x = x.transpose(1, 2).reshape(
#             x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
#         )

#         # print('SAHPEEE LASTT!', x.shape)
#         mask_logits = torch.einsum(
#             "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
#         )

#         return mask_logits, class_logits

#     @torch.compiler.disable
#     def _disable_attn_mask(self, attn_mask, prob):
#         if prob < 1:
#             random_queries = (
#                 torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
#                 > prob
#             )
#             attn_mask[
#                 :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
#             ][random_queries] = True

#         return attn_mask

#     def _attn(
#         self,
#         module: nn.Module,
#         x: torch.Tensor,
#         mask: Optional[torch.Tensor],
#         rope: Optional[torch.Tensor],
#     ):
#         if rope is not None:
#             if mask is not None:
#                 mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
#             return module(x, mask, rope)[0]

#         B, N, C = x.shape

#         qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
#         q, k = module.q_norm(q), module.k_norm(k)

#         if mask is not None:
#             mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

#         dropout_p = module.attn_drop.p if self.training else 0.0

#         if module.fused_attn:
#             x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
#         else:
#             attn = (q @ k.transpose(-2, -1)) * module.scale
#             if mask is not None:
#                 attn = attn.masked_fill(~mask, float("-inf"))
#             attn = F.softmax(attn, dim=-1)
#             attn = module.attn_drop(attn)
#             x = attn @ v

#         x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

#         return x

#     def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
#         attn_mask = torch.ones(
#             x.shape[0],
#             x.shape[1],
#             x.shape[1],
#             dtype=torch.bool,
#             device=x.device,
#         )
#         interpolated = F.interpolate(
#             mask_logits,
#             self.encoder.backbone.patch_embed.grid_size,
#             mode="bilinear",
#         )
#         interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
#         attn_mask[
#             :,
#             : self.num_q,
#             self.num_q + self.encoder.backbone.num_prefix_tokens :,
#         ] = (
#             interpolated > 0
#         )
#         attn_mask = self._disable_attn_mask(
#             attn_mask,
#             self.attn_mask_probs[
#                 i - len(self.encoder.backbone.blocks) + self.num_blocks
#             ],
#         )
#         return attn_mask

#     def _merge_by_info(self, x: torch.Tensor, merge_info: dict, mode: str = "mean") -> torch.Tensor:
#         if merge_info.get("type") != "bipartite":
#             raise ValueError("Unknown merge info for merge projection.")
#         unm_idx = merge_info["unm_idx"]
#         src_idx = merge_info["src_idx"]
#         dst_idx = merge_info["dst_idx"]

#         src, dst = x[..., ::2, :], x[..., 1::2, :]
#         n, t1, c = src.shape
#         r = src_idx.shape[1]
#         if r == 0:
#             return x
#         unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
#         src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
#         return torch.cat([unm, dst], dim=1)

#     def _project_mask_probs(self, mask_probs: torch.Tensor, merge_history: list) -> torch.Tensor:
#         if not merge_history:
#             return mask_probs
#         probs = mask_probs.transpose(1, 2)  # [B, N, Q]
#         for merge_info in merge_history:
#             probs = self._merge_by_info(probs, merge_info, mode="mean")
#         return probs.transpose(1, 2)

#     def _project_mask_probs_local(
#         self,
#         mask_probs: torch.Tensor,
#         local_merge_source: torch.Tensor,
#         has_queries: bool = True,
#     ) -> torch.Tensor:
#         if local_merge_source is None:
#             return mask_probs
#         protected = (self.num_q if has_queries else 0) + self.encoder.backbone.num_prefix_tokens
#         source_vis = local_merge_source[:, protected:, protected:]  # [B, M, N]
#         if source_vis.numel() == 0:
#             return mask_probs
#         s = source_vis.float()
#         denom = s.sum(dim=2, keepdim=False).clamp(min=1.0)  # [B, M]
#         # mask_probs: [B, Q, N] -> [B, Q, M]
#         merged = torch.einsum("bqn,bmn->bqm", mask_probs, s)
#         merged = merged / denom.unsqueeze(1)
#         return merged

#     def _merge_tokens(
#         self,
#         x: torch.Tensor,
#         layer_idx: int,
#         merge_history: Optional[list] = None,
#         merge_strategy: str = "project",
#         local_merge_source: Optional[torch.Tensor] = None,
#     ):
#         # Compute token merge masks. If we already merged, unmerge for mask prediction
#         # but keep using the current merged tokens for the new merge.
#         x_for_merge = x
#         if merge_history or local_merge_source is not None:
#             x_unmerged = x
#             if merge_history:
#                 for merged_mask in reversed(merge_history):
#                     x_unmerged = self._unmerge_tokens(x_unmerged, merged_mask)
#             if local_merge_source is not None:
#                 x_unmerged = self._algm_unmerge(x_unmerged, local_merge_source, has_queries=True)
#             mask_logits, _ = self._predict(self.encoder.backbone.norm(x_unmerged)) # [B, N, 128, 128]
#             if merge_strategy == "rematch":
#                 x_for_merge = x_unmerged
#         else:
#             mask_logits, _ = self._predict(self.encoder.backbone.norm(x)) # [B, N, 128, 128]

#         grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
#         interpolated = F.interpolate(
#             mask_logits,
#             (grid_h, grid_w),
#             mode="bilinear",
#         ) # [B, N, 32, 32]
#         vision_token_masks = torch.softmax(interpolated, dim=1).flatten(2)
#         if merge_history and merge_strategy == "project":
#             vision_token_masks = self._project_mask_probs(vision_token_masks, merge_history)
#             protected = self.num_q + self.encoder.backbone.num_prefix_tokens
#             vision_len = x.shape[1] - protected
#             if not dynamo.is_compiling():
#                 if vision_token_masks.shape[-1] != vision_len:
#                     warnings.warn(
#                         f"Projected mask length {vision_token_masks.shape[-1]} "
#                         f"!= current vision length {vision_len} at layer {layer_idx}."
#                     )
#                 assert vision_token_masks.shape[-1] == vision_len, (
#                     f"Mask projection mismatch at layer {layer_idx}: "
#                     f"{vision_token_masks.shape[-1]} vs {vision_len}"
#                 )

#         protected = self.num_q + self.encoder.backbone.num_prefix_tokens
#         vision_tokens = x_for_merge[:, protected:, :]
#         if local_merge_source is not None and vision_token_masks.shape[-1] != vision_tokens.shape[1]:
#             vision_token_masks = self._project_mask_probs_local(
#                 vision_token_masks, local_merge_source, has_queries=True
#             )
#             if not dynamo.is_compiling():
#                 if vision_token_masks.shape[-1] != vision_tokens.shape[1]:
#                     warnings.warn(
#                         f"Local mask projection mismatch {vision_token_masks.shape[-1]} "
#                         f"!= merged vision length {vision_tokens.shape[1]} at layer {layer_idx}."
#                     )
#                 assert vision_token_masks.shape[-1] == vision_tokens.shape[1], (
#                     f"Local mask projection mismatch at layer {layer_idx}: "
#                     f"{vision_token_masks.shape[-1]} vs {vision_tokens.shape[1]}"
#                 )
#         merge, merge_info = global_bipartite_matching_masked(
#             vision_tokens,
#             vision_token_masks,
#             layer_idx=layer_idx,
#             tau=self.merge_conf_threshold,
#             # score_metric=self.global_merge_score_metric,
#         )
#         merged_vision = merge(vision_tokens)
#         x = torch.cat((x_for_merge[:, :protected, :], merged_vision), dim=1)
#         if self.merge_logging_enabled and not dynamo.is_compiling():
#             warnings.warn(
#                 f"[global-merge] layer={layer_idx} vision_tokens={vision_tokens.shape[1]} "
#                 f"merged_vision={merged_vision.shape[1]} total_tokens={x.shape[1]}"
#             )

#         if merge_info is None:
#             return x, None

#         merge_info["type"] = "bipartite"
#         return x, merge_info

#     def _unmerge_tokens(self, x: torch.Tensor, mask):
#         # unmerge tokens
#         if isinstance(mask, dict):
#             if mask.get("type") != "bipartite":
#                 raise ValueError("Unknown merge info for unmerge.")
#             unm_idx = mask["unm_idx"]
#             src_idx = mask["src_idx"]
#             dst_idx = mask["dst_idx"]
#             num_patch = mask["num_patch"]

#             protected = self.num_q + self.encoder.backbone.num_prefix_tokens
#             merged = x[:, protected:, :]
#             if merged.shape[1] == num_patch:
#                 return x

#             unm_len = unm_idx.shape[1]
#             dst = merged[:, unm_len:, :]
#             unm = merged[:, :unm_len, :]
#             n, _, c = merged.shape
#             r = src_idx.shape[1]

#             src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
#             out_patch = torch.zeros(n, num_patch, c, device=x.device, dtype=x.dtype)
#             out_patch[..., 1::2, :] = dst
#             out_patch.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
#             out_patch.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
#             reprojected_tokens = out_patch
#         else:
#             fused = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
#             fused_expanded = fused.unsqueeze(2)
#             mask_expanded = mask.unsqueeze(-1)
#             reproj_sum = (mask_expanded * fused_expanded).sum(dim=1)
#             reproj_count = mask.sum(dim=1, keepdim=False).clamp(min=1e-6).unsqueeze(-1)  # [B, HW, 1]
#             reprojected_tokens = reproj_sum / reproj_count

#         x = torch.cat(
#             (x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], reprojected_tokens), dim=1
#         )
#         return x
    
#     def _algm_unmerge(self, x, source, has_queries=True):
#         # Keep this path allocation-free: infer whether query tokens are present
#         # in source/x and adjust protected offsets instead of padding/cropping tensors.
#         source_has_queries = bool(has_queries)
#         if source.shape[1] != x.shape[1]:
#             delta = source.shape[1] - x.shape[1]
#             if delta == self.num_q:
#                 source_has_queries = True
#             elif delta == -self.num_q:
#                 source_has_queries = False
#             else:
#                 # Fallback: keep current query alignment when mismatch is not +/-num_q.
#                 source_has_queries = bool(has_queries)

#         # assert False, f'{source.shape[1]} {x.shape[1]}'
#         source_protected = self.encoder.backbone.num_prefix_tokens + (
#             self.num_q if source_has_queries else 0
#         )
#         x_protected = self.encoder.backbone.num_prefix_tokens + (
#             self.num_q if has_queries else 0
#         )

#         target_vision_len = (
#             self.encoder.backbone.patch_embed.grid_size[0]
#             * self.encoder.backbone.patch_embed.grid_size[1]
#         )
#         # Rows = merged tokens; Cols = original tokens (before merges).
#         src_vis = source[:, source_protected:, source_protected:]
#         vision = x[:, x_protected:, :]

#         if src_vis.shape[2] > target_vision_len:
#             extra_cols = src_vis.shape[2] - target_vision_len
#             if extra_cols <= self.num_q:
#                 # Query spill is prepended in source maps; drop leading cols.
#                 src_vis = src_vis[:, :, extra_cols:]
#             else:
#                 raise ValueError(
#                     "Source has more original-token cols than expected. "
#                     f"{src_vis.shape[2]} > {target_vision_len} (extra={extra_cols})"
#                 )
#         elif src_vis.shape[2] < target_vision_len:
#             raise ValueError(
#                 "Source has fewer original-token cols than expected. "
#                 f"{src_vis.shape[2]} < {target_vision_len}"
#             )

#         if src_vis.shape[1] > vision.shape[1]:
#             extra_rows = src_vis.shape[1] - vision.shape[1]
#             if extra_rows <= self.num_q:
#                 # Query spill is prepended in source maps; drop leading rows.
#                 src_vis = src_vis[:, extra_rows:, :]
#             else:
#                 raise ValueError(
#                     "Source has more merged tokens than x has vision tokens. "
#                     f"{src_vis.shape[1]} > {vision.shape[1]} (extra={extra_rows})"
#                 )

#         idxs = src_vis.argmax(dim=1)  # [B, original_vision_len]
#         if idxs.shape[1] != target_vision_len:
#             raise ValueError(
#                 "Number of original vision tokens does not match target. "
#                 f"{idxs.shape[1]} != {target_vision_len}"
#             )

#         x_ = torch.gather(
#             vision,
#             1,
#             idxs.unsqueeze(-1).expand(-1, idxs.shape[1], vision.shape[2]),
#         )

#         x = torch.cat((x[:, :x_protected, :], x_), dim=1)

#         return x

#     # def _algm_unmerge(self, x: torch.Tensor, source: torch.Tensor, has_queries: bool = True):
#     #     GS = self.encoder.backbone.patch_embed.grid_size[0]
#     #     # x = self.head(x)
#     #     if source is not None :
            
#     #         idxs = source[:, self.encoder.backbone.num_prefix_tokens:, self.encoder.backbone.num_prefix_tokens:].argmax(dim=1)
#     #         # defining a torch tensor with the output size 
#     #         masks_shape = x.shape
#     #         size_x = source.shape[2]-self.encoder.backbone.num_prefix_tokens
#     #         x_ =  torch.ones(masks_shape[0],size_x,masks_shape[2],device=x.device)
        
#     #         print('LASTTT!!',x_.shape, x.shape, idxs.shape)
#     #         assert False, f'LASTTT!! {x_.shape}, {x.shape}, {idxs.shape}'
#     #         for batch in range(0,masks_shape[0]):
#     #             x_[batch,:,:] = x[batch,idxs[batch]]

#     #             # x_ = torch.cat((x[:, :self.encoder.backbone.num_prefix_tokens, :], x_), dim=1)
#     #             # x_ = torch.cat((source[:]))
            
#     #         x_ = torch.cat((x[:, :self.encoder.backbone.num_prefix_tokens, :], x_), dim=1)
#     #     else:
#     #             x_ = x

#     #     # x_ = rearrange(x_, "b (h w) c -> b c h w", h=GS)
#     #     return x_

#     def forward(
#         self,
#         x: torch.Tensor,
#     ):
#         x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

#         # self._reset_turbo_info()
#         x = self.encoder.backbone.patch_embed(x)

#         if hasattr(self.encoder.backbone, "_pos_embed"):
#             x = self.encoder.backbone._pos_embed(x)

#         mask_logits_per_layer, class_logits_per_layer = [], []
#         turbo_info = getattr(self.encoder.backbone, "_turbo_info", None)
#         num_prefix_tokens = self.encoder.backbone.num_prefix_tokens
#         num_protected_tokens = num_prefix_tokens
#         target_vision_len = (
#             self.encoder.backbone.patch_embed.grid_size[0]
#             * self.encoder.backbone.patch_embed.grid_size[1]
#         )
#         source = turbo_info["source"] if turbo_info is not None else None
#         merged_masks = []
#         local_merge_source = None
#         local_merge_size = None
#         local_merged = False
#         # merge_after_block_idx = 20  # 0-based index; adjust if you want a different layer.

#         has_queries = False
#         queries_persistent = False
#         if self.query_insert_layer is None:
#             query_insert_layer = len(self.encoder.backbone.blocks) - self.num_blocks
#         else:
#             query_insert_layer = max(int(self.query_insert_layer) - 1, 0)
#             query_insert_layer = min(query_insert_layer, len(self.encoder.backbone.blocks) - 1)
#         if self.query_reinsert_layer is None:
#             query_reinsert_layer = len(self.encoder.backbone.blocks) - self.num_blocks
#         else:
#             query_reinsert_layer = max(int(self.query_reinsert_layer) - 1, 0)
#             query_reinsert_layer = min(query_reinsert_layer, len(self.encoder.backbone.blocks) - 1)

#         def _prepend_queries(
#             x_tokens: torch.Tensor,
#             source_map: Optional[torch.Tensor],
#         ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
#             if local_merged and source_map is not None and self.num_q > 0:
#                 expected_without_queries = x_tokens.shape[1]
#                 expected_with_queries = expected_without_queries + self.num_q
#                 expected_cols_without_queries = num_prefix_tokens + target_vision_len
#                 expected_cols_with_queries = expected_cols_without_queries + self.num_q
#                 rows = source_map.shape[1]
#                 cols = source_map.shape[2]
#                 if rows == expected_without_queries:
#                     if cols == expected_cols_without_queries:
#                         padded_source = source_map.new_zeros(
#                             source_map.shape[0],
#                             expected_with_queries,
#                             expected_cols_with_queries,
#                         )
#                         padded_source[:, self.num_q :, self.num_q :] = source_map
#                         diag_idx = torch.arange(self.num_q, device=source_map.device)
#                         padded_source[:, diag_idx, diag_idx] = 1
#                         source_map = padded_source
#                     elif cols == expected_cols_with_queries:
#                         # Cols already include query space; prepend only rows.
#                         padded_source = source_map.new_zeros(
#                             source_map.shape[0],
#                             expected_with_queries,
#                             expected_cols_with_queries,
#                         )
#                         padded_source[:, self.num_q :, :] = source_map
#                         diag_idx = torch.arange(self.num_q, device=source_map.device)
#                         padded_source[:, diag_idx, diag_idx] = 1
#                         source_map = padded_source
#                     else:
#                         raise ValueError(
#                             "Unexpected local_merge_source cols before query prepend. "
#                             f"cols={cols}, expected_no_q={expected_cols_without_queries}, "
#                             f"expected_with_q={expected_cols_with_queries}"
#                         )
#                 elif rows == expected_with_queries:
#                     # Already query-aligned in rows; ensure query cols are present.
#                     if cols == expected_cols_without_queries:
#                         padded_source = source_map.new_zeros(
#                             source_map.shape[0],
#                             expected_with_queries,
#                             expected_cols_with_queries,
#                         )
#                         padded_source[:, :, self.num_q :] = source_map
#                         diag_idx = torch.arange(self.num_q, device=source_map.device)
#                         padded_source[:, diag_idx, diag_idx] = 1
#                         source_map = padded_source
#                     elif cols != expected_cols_with_queries:
#                         raise ValueError(
#                             "Unexpected local_merge_source cols in query-on state. "
#                             f"cols={cols}, expected_with_q={expected_cols_with_queries}"
#                         )
#                 else:
#                     raise ValueError(
#                         "Unexpected local_merge_source shape before query prepend. "
#                         f"source_map={tuple(source_map.shape[1:])}, "
#                         f"expected_without_queries={expected_without_queries}, "
#                         f"expected_with_queries={expected_with_queries}"
#                     )
#             x_tokens = torch.cat(
#                 (self.q.weight[None, :, :].expand(x_tokens.shape[0], -1, -1), x_tokens), dim=1
#             )
#             return x_tokens, source_map

#         for i, block in enumerate(self.encoder.backbone.blocks):
#             # if i>=8:
#             #     assert False, 'Local merge should be done after 1'
#             if i == query_insert_layer and not has_queries:
#                 x, local_merge_source = _prepend_queries(x, local_merge_source)
#                 has_queries = True
#                 queries_persistent = i == query_reinsert_layer
#             elif i == query_reinsert_layer and not has_queries:
#                 x, local_merge_source = _prepend_queries(x, local_merge_source)
#                 has_queries = True
#                 queries_persistent = True
#                 # if self.encoder.backbone._turbo_info["source"] is not None:
#                     # x = self._algm_unmerge(x, self.encoder.backbone._turbo_info["source"], has_queries=False)
#                     # assert False
#                 # num_protected_tokens += self.num_q
#                 # target_tokens = x.shape[1] + self.num_q
#                 # if source is not None and source.shape[1] != target_tokens:
#                 #     # Expand source to include queries: rows and cols grow by num_q
#                 #     padded_source = source.new_zeros(
#                 #         source.shape[0],
#                 #         source.shape[1] + self.num_q,
#                 #         source.shape[2] + self.num_q,
#                 #     )
#                 #     padded_source[:, self.num_q :, self.num_q :] = source
#                 #     if self.num_q > 0:
#                 #         diag_idx = torch.arange(self.num_q, device=source.device)
#                 #         padded_source[:, diag_idx, diag_idx] = 1
#                 #     turbo_info["source"] = padded_source
#                 #     source = padded_source
#                 # print('SAHPEEE!', x.shape)

#             if hasattr(block, "_turbo_info"):
#                 # print('i:', i)
#                 x = block(x, num_protected_tokens=num_protected_tokens)
#                 # source = self.encoder.backbone._turbo_info["source"]
#             else:
#                 x = block(x)
#             if (
#                 self.local_merge_enabled
#                 and not has_queries
#                 and not local_merged
#                 and (i + 1) == max(int(self.local_merge_after_layer), 1)
#             ):
#                 before_tokens = x.shape[1]
#                 merge = conditional_pooling(
#                     x,
#                     self.local_merge_threshold,
#                     self.local_merge_window_size,
#                     num_prefix_tokens=num_protected_tokens,
#                 )
#                 local_merge_source = merge_source(merge, x, local_merge_source)
#                 x, local_merge_size = merge_wavg(merge, x, local_merge_size)
#                 local_merged = True
#                 if self.merge_logging_enabled and not dynamo.is_compiling():
#                     warnings.warn(
#                         f"[local-merge] layer={i + 1} tokens={before_tokens}->{x.shape[1]}"
#                     )
#             if has_queries and i < len(self.encoder.backbone.blocks) - 1 and self._should_merge(i + 1):
#                 x, merged_mask = self._merge_tokens(
#                     x,
#                     layer_idx=i + 1,
#                     merge_history=merged_masks,
#                     merge_strategy=self.merge_strategy,
#                     local_merge_source=local_merge_source,
#                 )
#                 # Keep local source row-space synchronized with current x after
#                 # each global merge; this avoids stale row-count mismatches later.
#                 if local_merge_source is not None and merged_mask is not None:
#                     local_protected = self.num_q + num_prefix_tokens
#                     if local_merge_source.shape[1] >= local_protected:
#                         local_prefix = local_merge_source[:, :local_protected, :]
#                         local_vis = local_merge_source[:, local_protected:, :]
#                         local_vis = self._merge_by_info(local_vis, merged_mask, mode="amax")
#                         local_merge_source = torch.cat((local_prefix, local_vis), dim=1)
#                 if self.merge_strategy == "rematch" and local_merge_source is not None:
#                     local_merged = False
#                     local_merge_source = None
#                     local_merge_size = None
#                 if merged_mask is not None:
#                     if self.merge_strategy == "rematch":
#                         merged_masks = [merged_mask]
#                     else:
#                         merged_masks.append(merged_mask)
#                 elif self.merge_strategy == "rematch":
#                     merged_masks = []
#             if has_queries and not queries_persistent and i == query_insert_layer:
#                 x = x[:, self.num_q :, :]
#                 if local_merged and local_merge_source is not None and self.num_q > 0:
#                     expected_cols_with_queries = num_prefix_tokens + self.num_q + target_vision_len
#                     if local_merge_source.shape[1] >= self.num_q:
#                         local_merge_source = local_merge_source[:, self.num_q :, :]
#                     if (
#                         local_merge_source.shape[2] >= self.num_q
#                         and local_merge_source.shape[2] == expected_cols_with_queries
#                     ):
#                         local_merge_source = local_merge_source[:, :, self.num_q :]
#                 has_queries = False
#             # if i == len(self.encoder.backbone.blocks) - self.num_blocks:
#             #     x, merged_mask = self._merge_tokens(x, layer_idx=i + 1)
#         # source = turbo_info["source"]
#         #### This is code for ALGM token merging baseline. ######
#         # if self.encoder.backbone._turbo_info["source"] is not None:
#         #     x_vis = x[:, self.num_q:, :]
#         #     x_vis = self._algm_unmerge(x_vis, self.encoder.backbone._turbo_info["source"], has_queries=False)
#         #     x = torch.cat((x[:, :self.num_q, :], x_vis), dim=1)
#         #### End of ALGM token merging baseline. ######

#         if merged_masks or local_merged:
#             for merged_mask in reversed(merged_masks):
#                 x = self._unmerge_tokens(x, merged_mask)
#             if local_merged and local_merge_source is not None:
#                 x = self._algm_unmerge(x, local_merge_source, has_queries=has_queries)
#             if not dynamo.is_compiling():
#                 expected_vision_len = (
#                     self.encoder.backbone.patch_embed.grid_size[0]
#                     * self.encoder.backbone.patch_embed.grid_size[1]
#                 )
#                 protected = (self.num_q if has_queries else 0) + self.encoder.backbone.num_prefix_tokens
#                 actual_vision_len = x.shape[1] - protected
#                 if self.merge_logging_enabled:
#                     warnings.warn(
#                         f"[unmerge] vision_tokens={actual_vision_len} expected={expected_vision_len} "
#                         f"protected={protected} total_tokens={x.shape[1]}"
#                     )
#                 if actual_vision_len != expected_vision_len:
#                     warnings.warn(
#                         f"Unmerge produced {actual_vision_len} vision tokens, "
#                         f"expected {expected_vision_len}."
#                     )
#                 assert actual_vision_len == expected_vision_len, (
#                     f"Unmerge vision token mismatch: {actual_vision_len} vs {expected_vision_len}"
#                 )

#         mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
#         mask_logits_per_layer.append(mask_logits)
#         class_logits_per_layer.append(class_logits)

#         return (
#             mask_logits_per_layer,
#             class_logits_per_layer,
#         )









##### Old Code #########

# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import torch._dynamo as dynamo
from timm.layers import DropPath
from timm.layers import resample_abs_pos_embed

from models.scale_block import ScaleBlock
from algm_addition import apply_patch
from einops import rearrange
from models.algm_layers.global_merge_bavesh import global_bipartite_matching_masked
from models.algm_layers.local_merge import conditional_pooling, merge_source, merge_wavg
# from models.algm_layers.apt_entropy import (
#     build_base_keep_mask_from_entropy,
#     build_base_keep_mask_from_entropy_maps,
#     compute_patch_entropy_batched,
#     select_patches_by_threshold,
# )

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


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
        merge_start_layer: int = 16,
        merge_interval: int = 4,
        query_insert_layer: int = 16,
        query_reinsert_layer: int = 20,
        merge_conf_threshold_end: float = 0.0,
        merge_schedule_epochs: int = 7,
        merge_schedule_enabled: bool = True,
        global_merge_min_ratio: float = 0.85,
        global_merge_score_metric: str = "js",
        merge_strategy: str = "project",
        local_merge_enabled: bool = True,
        local_merge_after_layer: int = 1,
        local_merge_window_size: tuple[int, int] = (2, 2),
        local_merge_threshold: float = 0.88,
        merge_logging_enabled: bool = False,
        entropy_tokenization_enabled: bool = False,
        entropy_num_scales: int = 2,
        entropy_thresholds: Optional[List[float]] = None,
        entropy_bins: int = 64,
        entropy_tokenization_mode: str = "hybrid",
        entropy_merge_channel_weight: float = 1.0,
        entropy_keep_channel_weight: float = 1.0,
        entropy_pre_transformer_merge_enabled: bool = False,
        entropy_varlen_flash_enabled: bool = False,
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
        self.global_merge_min_ratio = float(global_merge_min_ratio)
        self.global_merge_score_metric = str(global_merge_score_metric)
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
        self.entropy_tokenization_enabled = bool(entropy_tokenization_enabled)
        self.entropy_num_scales = int(entropy_num_scales)
        self.entropy_thresholds = entropy_thresholds
        self.entropy_bins = int(entropy_bins)
        self.entropy_tokenization_mode = str(entropy_tokenization_mode)
        self.entropy_merge_channel_weight = float(entropy_merge_channel_weight)
        self.entropy_keep_channel_weight = float(entropy_keep_channel_weight)
        self.entropy_pre_transformer_merge_enabled = bool(entropy_pre_transformer_merge_enabled)
        self.entropy_varlen_flash_enabled = bool(entropy_varlen_flash_enabled)
        if self.merge_strategy not in ("project", "rematch"):
            raise ValueError("merge_strategy must be one of: 'project', 'rematch'")
        if not (0.0 <= self.global_merge_min_ratio <= 1.0):
            raise ValueError("global_merge_min_ratio must be in [0, 1].")
        if self.global_merge_score_metric not in ("cosine", "js", "hellinger", "dot"):
            raise ValueError(
                "global_merge_score_metric must be one of: 'cosine', 'js', 'hellinger', 'dot'"
            )
        if self.entropy_tokenization_mode not in ("hybrid", "entropy_only"):
            raise ValueError("entropy_tokenization_mode must be one of: 'hybrid', 'entropy_only'")
        if self.entropy_num_scales < 1:
            raise ValueError("entropy_num_scales must be >= 1")

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
        # self.entropy_patch_attn = nn.Conv2d(
        #     self.encoder.backbone.embed_dim,
        #     self.encoder.backbone.embed_dim,
        #     kernel_size=2,
        #     stride=2,
        # )
        # self.entropy_zero_mlp = nn.Linear(
        #     self.encoder.backbone.embed_dim,
        #     self.encoder.backbone.embed_dim,
        # )
        # nn.init.zeros_(self.entropy_zero_mlp.weight)
        # nn.init.zeros_(self.entropy_zero_mlp.bias)

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

    def _merge_predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        # class_logits = self.class_head(q)

        # assert False, f'SAHPEEE! , {x.shape}, {self.encoder.backbone.num_prefix_tokens}'
        # print(f'SAHPEEE! , {x.shape}, {self.encoder.backbone.num_prefix_tokens}')
        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        # print('SAHPEEE LASTT!', x.shape)
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), x
        )

        return mask_logits

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

    def _project_token_values(self, token_values: torch.Tensor, merge_history: list) -> torch.Tensor:
        if not merge_history:
            return token_values
        values = token_values.unsqueeze(-1)  # [B, N, 1]
        for merge_info in merge_history:
            values = self._merge_by_info(values, merge_info, mode="mean")
        return values.squeeze(-1)

    def _project_token_values_local(
        self,
        token_values: torch.Tensor,
        local_merge_source: torch.Tensor,
        has_queries: bool = True,
    ) -> torch.Tensor:
        if local_merge_source is None:
            return token_values
        protected = (self.num_q if has_queries else 0) + self.encoder.backbone.num_prefix_tokens
        source_vis = local_merge_source[:, protected:, protected:]  # [B, M, N]
        if source_vis.numel() == 0:
            return token_values
        s = source_vis.float()
        denom = s.sum(dim=2, keepdim=False).clamp(min=1.0)  # [B, M]
        merged = torch.einsum("bn,bmn->bm", token_values, s)
        merged = merged / denom
        return merged

    def _compute_entropy_keep_mask(self, raw_images: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.entropy_tokenization_enabled:
            return None
        patch_size = self.encoder.backbone.patch_embed.patch_size
        base_patch_size = int(max(patch_size[0], patch_size[1]))
        keep = build_base_keep_mask_from_entropy(
            raw_images=raw_images,
            base_patch_size=base_patch_size,
            num_scales=self.entropy_num_scales,
            thresholds=self.entropy_thresholds,
            bins=self.entropy_bins,
        )
        return keep

    def _compute_entropy_maps(
        self,
        raw_images: torch.Tensor,
        entropy_maps: Optional[dict[int, torch.Tensor]] = None,
    ) -> Optional[dict[int, torch.Tensor]]:
        if not self.entropy_tokenization_enabled:
            return None
        if entropy_maps is not None:
            return entropy_maps
        patch_size = self.encoder.backbone.patch_embed.patch_size
        base_patch_size = int(max(patch_size[0], patch_size[1]))
        images = raw_images
        if images.dtype.is_floating_point:
            max_val = float(images.detach().amax().item())
            if max_val <= 1.5:
                images = images * 255.0
        images = images.clamp(0.0, 255.0)
        return compute_patch_entropy_batched(
            images=images,
            patch_size=base_patch_size,
            num_scales=self.entropy_num_scales,
            bins=self.entropy_bins,
        )

    def _compute_entropy_keep_mask_from_maps(
        self,
        entropy_maps: Optional[dict[int, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if not self.entropy_tokenization_enabled or entropy_maps is None:
            return None
        patch_size = self.encoder.backbone.patch_embed.patch_size
        base_patch_size = int(max(patch_size[0], patch_size[1]))
        return build_base_keep_mask_from_entropy_maps(
            entropy_maps=entropy_maps,
            base_patch_size=base_patch_size,
            num_scales=self.entropy_num_scales,
            thresholds=self.entropy_thresholds,
        )

    def _compute_entropy_masks_from_maps(
        self,
        entropy_maps: Optional[dict[int, torch.Tensor]],
    ) -> Optional[dict[int, torch.Tensor]]:
        if not self.entropy_tokenization_enabled or entropy_maps is None:
            return None
        patch_size = self.encoder.backbone.patch_embed.patch_size
        base_patch_size = int(max(patch_size[0], patch_size[1]))
        patch_sizes = [base_patch_size * (2**i) for i in range(self.entropy_num_scales)]
        normalized_maps: dict[int, torch.Tensor] = {}
        for ps in patch_sizes:
            if ps not in entropy_maps:
                continue
            emap = entropy_maps[ps]
            if emap.dim() == 2:
                emap = emap.unsqueeze(0)
            normalized_maps[ps] = emap
        if base_patch_size not in normalized_maps:
            return None
        thresholds = self.entropy_thresholds
        if thresholds is None:
            thresholds = [4.5] * max(0, self.entropy_num_scales - 1)
        return select_patches_by_threshold(normalized_maps, thresholds)

    def _entropy_pre_transformer_merge(
        self,
        x: torch.Tensor,
        entropy_keep_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """
        Merge base patch tokens before transformer blocks using entropy keep mask.
        Returns merged tokens + source map for exact unmerge.
        """
        if not self.entropy_pre_transformer_merge_enabled or entropy_keep_mask is None:
            return x, None, False

        protected = self.encoder.backbone.num_prefix_tokens
        vision = x[:, protected:, :]
        bsz, n_tokens, dim = vision.shape
        grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
        if grid_h * grid_w != n_tokens:
            return x, None, False
        if entropy_keep_mask.shape != (bsz, n_tokens):
            return x, None, False

        if self.entropy_num_scales < 2:
            return x, None, False

        # One pre-transformer coarse merge stage (2x2).
        # APT-style token construction:
        #   coarse_token = avg_pool(base_tokens_2x2) + zero_mlp(conv2x2(base_token_grid))
        # Merge locations are selected per-image. For dense batched execution, we keep
        # a shared merged-token count R = min(mergeable_count_per_image).
        keep_bool = entropy_keep_mask.view(bsz, grid_h, grid_w) > 0.5
        non_keep = ~keep_bool
        if grid_h % 2 != 0 or grid_w % 2 != 0:
            return x, None, False

        mergeable = (
            non_keep.unfold(1, 2, 2)
            .unfold(2, 2, 2)
            .all(dim=-1)
            .all(dim=-1)
        )  # [B, H/2, W/2]
        h2, w2 = grid_h // 2, grid_w // 2
        mergeable_flat = mergeable.flatten(1)  # [B, H/2*W/2]
        merge_counts = mergeable_flat.sum(dim=1)
        r = int(merge_counts.min().item())
        if r <= 0:
            return x, None, False

        vision_grid = vision.transpose(1, 2).reshape(bsz, dim, grid_h, grid_w)
        pooled_grid = F.avg_pool2d(vision_grid, kernel_size=2, stride=2)
        attn_grid = self.entropy_patch_attn(vision_grid)
        attn_tokens = attn_grid.flatten(2).transpose(1, 2)
        delta_tokens = self.entropy_zero_mlp(attn_tokens)
        coarse_tokens = pooled_grid.flatten(2).transpose(1, 2) + delta_tokens  # [B, H/2*W/2, C]

        token_idx_grid = torch.arange(n_tokens, device=x.device).view(grid_h, grid_w)
        coarse_to_tokens = []
        for block_row in range(h2):
            for block_col in range(w2):
                ys = 2 * block_row
                xs = 2 * block_col
                block_ids = token_idx_grid[ys : ys + 2, xs : xs + 2].reshape(-1)
                coarse_to_tokens.append(block_ids)
        coarse_to_tokens = torch.stack(coarse_to_tokens, dim=0)  # [H/2*W/2, 4]

        selected_coarse_idx = torch.empty((bsz, r), device=x.device, dtype=torch.long)
        for bi in range(bsz):
            cand = torch.nonzero(mergeable_flat[bi], as_tuple=False).squeeze(1)
            selected_coarse_idx[bi] = cand[:r]

        selected_rows = coarse_to_tokens.index_select(0, selected_coarse_idx.flatten()).view(bsz, r, 4)
        merged_token_mask = torch.zeros((bsz, n_tokens), dtype=torch.bool, device=x.device)
        merged_token_mask.scatter_(1, selected_rows.reshape(bsz, -1), True)
        remaining_ids_full = torch.arange(n_tokens, device=x.device).unsqueeze(0).expand(bsz, -1)
        remaining_ids = remaining_ids_full[~merged_token_mask].view(bsz, n_tokens - (4 * r))

        merged_coarse_tokens = coarse_tokens.gather(
            1,
            selected_coarse_idx.unsqueeze(-1).expand(-1, -1, dim),
        )
        num_merged = merged_coarse_tokens.shape[1]
        num_remaining = remaining_ids.shape[1]
        out_tokens = num_merged + num_remaining

        source_vis = torch.zeros((bsz, out_tokens, n_tokens), device=x.device, dtype=x.dtype)
        if num_merged > 0:
            for bi in range(bsz):
                source_vis[bi, :num_merged].scatter_(
                    1,
                    selected_rows[bi],
                    torch.ones_like(selected_rows[bi], dtype=x.dtype),
                )
        if num_remaining > 0:
            batch_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand(-1, num_remaining)
            out_row_idx = (
                torch.arange(num_remaining, device=x.device).unsqueeze(0).expand(bsz, -1)
                + num_merged
            )
            source_vis[batch_idx, out_row_idx, remaining_ids] = 1.0

        remaining_tokens = (
            vision.gather(1, remaining_ids.unsqueeze(-1).expand(-1, -1, dim))
            if num_remaining > 0
            else vision[:, :0, :]
        )
        merged_vision = torch.cat((merged_coarse_tokens, remaining_tokens), dim=1)

        x_merged = torch.cat((x[:, :protected, :], merged_vision), dim=1)

        source_full = torch.zeros(
            bsz,
            protected + out_tokens,
            protected + n_tokens,
            device=x.device,
            dtype=x.dtype,
        )
        if protected > 0:
            eye = torch.eye(protected, device=x.device, dtype=x.dtype).unsqueeze(0).expand(bsz, -1, -1)
            source_full[:, :protected, :protected] = eye
        source_full[:, protected:, protected:] = source_vis

        return x_merged, source_full, True

    def _entropy_pre_transformer_tokenize_varlen(
        self,
        x_norm: torch.Tensor,
        entropy_maps: Optional[dict[int, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        APT-like per-image adaptive tokenization with multi-scale selected tokens.
        Uses:
          - entropy threshold selection per scale
          - per-scale resampled absolute positional embeddings
          - resized token branch + zero-MLP conv token branch for coarse tokens
        Returns:
            token_list: list of [Li, C] (includes backbone prefix tokens)
            source_vis_list: list of [Mi, N_base] mapping merged vision tokens to original base tokens
        """
        bsz = x_norm.shape[0]
        backbone = self.encoder.backbone
        protected = self.encoder.backbone.num_prefix_tokens
        base_patch_size = int(max(backbone.patch_embed.patch_size))
        grid_h, grid_w = backbone.patch_embed.grid_size
        dim = backbone.embed_dim
        n_tokens = grid_h * grid_w

        token_list: list[torch.Tensor] = []
        source_vis_list: list[torch.Tensor] = []

        masks = self._compute_entropy_masks_from_maps(entropy_maps)

        base_proj_grid = backbone.patch_embed.proj(x_norm)  # [B, D, H, W]
        if base_proj_grid.shape[-2:] != (grid_h, grid_w):
            raise ValueError(
                f"Patch grid mismatch: proj={tuple(base_proj_grid.shape[-2:])}, "
                f"backbone grid={(grid_h, grid_w)}"
            )
        base_embed = base_proj_grid.flatten(2).transpose(1, 2)  # [B, N, D]

        pos_embed = getattr(backbone, "pos_embed", None)
        if pos_embed is None:
            raise ValueError("APT-like tokenization requires absolute pos_embed on backbone.")
        base_pos = pos_embed
        # Some backbones expose num_prefix_tokens (e.g., cls + reg), but pos_embed
        # may only contain a subset of those prefix positions (commonly cls only).
        # Infer prefix token count from pos_embed length to avoid invalid resampling.
        pos_prefix_tokens = int(base_pos.shape[1] - n_tokens)
        if pos_prefix_tokens < 0:
            raise ValueError(
                f"pos_embed shorter than patch token count: {base_pos.shape[1]} < {n_tokens}"
            )
        if pos_prefix_tokens > protected:
            pos_prefix_tokens = protected

        if base_pos.shape[1] != n_tokens + pos_prefix_tokens:
            old_patch_tokens = base_pos.shape[1] - pos_prefix_tokens
            old_grid = int(round(old_patch_tokens ** 0.5))
            if old_grid * old_grid != old_patch_tokens:
                raise ValueError(
                    "Cannot infer old patch grid for pos_embed resampling: "
                    f"patch_tokens={old_patch_tokens}"
                )
            base_pos = resample_abs_pos_embed(
                base_pos,
                new_size=(grid_h, grid_w),
                old_size=(old_grid, old_grid),
                num_prefix_tokens=pos_prefix_tokens,
            )

        prefix_pos_src = (
            base_pos[:, :pos_prefix_tokens, :]
            if pos_prefix_tokens > 0
            else base_pos.new_zeros(1, 0, dim)
        )
        if protected == pos_prefix_tokens:
            prefix_pos = prefix_pos_src
        elif pos_prefix_tokens == 0:
            prefix_pos = base_pos.new_zeros(1, protected, dim)
        elif pos_prefix_tokens == 1:
            prefix_pos = prefix_pos_src.expand(-1, protected, -1)
        else:
            reps = (protected + pos_prefix_tokens - 1) // pos_prefix_tokens
            prefix_pos = prefix_pos_src.repeat(1, reps, 1)[:, :protected, :]

        base_patch_pos = base_pos[:, pos_prefix_tokens:, :].reshape(1, n_tokens, dim)

        cls_token = getattr(backbone, "cls_token", None)
        reg_token = getattr(backbone, "reg_token", None)
        if cls_token is not None:
            cls_token = cls_token.expand(bsz, -1, -1)
        if reg_token is not None:
            reg_token = reg_token.expand(bsz, -1, -1)

        if (
            (not self.entropy_pre_transformer_merge_enabled)
            or (masks is None)
            or (self.entropy_num_scales < 2)
        ):
            eye = torch.eye(n_tokens, device=x_norm.device, dtype=torch.bool)
            for bi in range(bsz):
                prefix_parts = []
                if cls_token is not None:
                    prefix_parts.append(cls_token[bi : bi + 1])
                if reg_token is not None:
                    prefix_parts.append(reg_token[bi : bi + 1])
                prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else base_embed[bi : bi + 1, :0, :]
                prefix = prefix + prefix_pos
                token_list.append(torch.cat((prefix[0], base_embed[bi] + base_patch_pos[0]), dim=0))
                source_vis_list.append(eye)
            return token_list, source_vis_list

        token_idx_grid = torch.arange(n_tokens, device=x_norm.device).view(grid_h, grid_w)
        eye = torch.eye(n_tokens, device=x_norm.device, dtype=torch.bool)
        pos_cache: dict[int, torch.Tensor] = {1: base_patch_pos}
        resized_embed_cache: dict[int, torch.Tensor] = {1: base_embed}
        coarse_source_cache: dict[int, torch.Tensor] = {}

        for scale_idx in range(1, self.entropy_num_scales):
            scale_factor = 2 ** scale_idx
            h_s = grid_h // scale_factor
            w_s = grid_w // scale_factor
            if h_s <= 0 or w_s <= 0:
                continue

            x_scaled = F.interpolate(
                x_norm,
                scale_factor=(1.0 / scale_factor),
                mode="bilinear",
                align_corners=False,
            )
            resized_embed_cache[scale_factor] = backbone.patch_embed.proj(x_scaled).flatten(2).transpose(1, 2)

            pos_resampled = resample_abs_pos_embed(
                base_pos,
                new_size=(h_s, w_s),
                old_size=(grid_h, grid_w),
                num_prefix_tokens=pos_prefix_tokens,
            )
            pos_cache[scale_factor] = pos_resampled[:, pos_prefix_tokens:, :]

            block_ids = (
                token_idx_grid.unfold(0, scale_factor, scale_factor)
                .unfold(1, scale_factor, scale_factor)
                .contiguous()
                .reshape(h_s * w_s, scale_factor * scale_factor)
            )
            source_template = torch.zeros(h_s * w_s, n_tokens, device=x_norm.device, dtype=torch.bool)
            source_template.scatter_(1, block_ids, True)
            coarse_source_cache[scale_factor] = source_template

        for bi in range(bsz):
            prefix_parts = []
            if cls_token is not None:
                prefix_parts.append(cls_token[bi : bi + 1])
            if reg_token is not None:
                prefix_parts.append(reg_token[bi : bi + 1])
            prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else base_embed[bi : bi + 1, :0, :]
            prefix = prefix + prefix_pos

            vis_tokens: list[torch.Tensor] = []
            source_rows: list[torch.Tensor] = []

            # Base scale selected fine tokens.
            base_mask = masks[base_patch_size][bi]
            base_h = min(base_mask.shape[0], grid_h)
            base_w = min(base_mask.shape[1], grid_w)
            base_mask = base_mask[:base_h, :base_w]
            base_flat_ids = torch.nonzero(base_mask.flatten() > 0.5, as_tuple=False).squeeze(1)
            if base_flat_ids.numel() > 0:
                vis_tokens.append(
                    base_embed[bi].index_select(0, base_flat_ids)
                    + base_patch_pos[0].index_select(0, base_flat_ids)
                )
                source_rows.append(eye.index_select(0, base_flat_ids))

            attn_grid_by_scale: dict[int, torch.Tensor] = {}
            attn_grid = base_proj_grid[bi : bi + 1]
            for scale_idx in range(1, self.entropy_num_scales):
                scale_factor = 2 ** scale_idx
                attn_grid = self.entropy_patch_attn(attn_grid)
                attn_grid_by_scale[scale_factor] = attn_grid

            # Coarser scales selected tokens.
            for scale_idx in range(1, self.entropy_num_scales):
                scale_factor = 2 ** scale_idx
                cur_ps = base_patch_size * scale_factor
                if cur_ps not in masks:
                    continue

                h_s = grid_h // scale_factor
                w_s = grid_w // scale_factor
                if h_s <= 0 or w_s <= 0:
                    continue

                resized_embed = resized_embed_cache[scale_factor]
                cur_pos = pos_cache[scale_factor]

                cur_mask = masks[cur_ps][bi]
                cur_mask = cur_mask[:h_s, :w_s]
                cur_flat = torch.nonzero(cur_mask.flatten() > 0.5, as_tuple=False).squeeze(1)
                if cur_flat.numel() == 0:
                    continue

                tok_delta_all = self.entropy_zero_mlp(
                    attn_grid_by_scale[scale_factor].flatten(2).transpose(1, 2).squeeze(0)
                )

                cur_tokens = (
                    resized_embed[bi].index_select(0, cur_flat)
                    + cur_pos[0].index_select(0, cur_flat)
                    + tok_delta_all.index_select(0, cur_flat)
                )

                vis_tokens.append(cur_tokens)
                source_rows.append(coarse_source_cache[scale_factor].index_select(0, cur_flat))

            if vis_tokens:
                merged_vision = torch.cat(vis_tokens, dim=0)
                source_vis = torch.cat(source_rows, dim=0)
            else:
                merged_vision = base_embed[bi : bi + 1, :0, :][0]
                source_vis = torch.zeros(0, n_tokens, device=x_norm.device, dtype=torch.bool)

            tokens = torch.cat((prefix[0], merged_vision), dim=0)
            token_list.append(tokens)
            source_vis_list.append(source_vis)

        return token_list, source_vis_list

    def _pack_token_list(self, token_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        seqlens = torch.tensor([t.shape[0] for t in token_list], dtype=torch.int32, device=token_list[0].device)
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=token_list[0].device), seqlens.cumsum(0)]
        )
        max_seqlen = int(seqlens.max().item())
        packed = torch.cat(token_list, dim=0)
        return packed, cu_seqlens, max_seqlen, seqlens

    def _unpack_token_list(self, packed: torch.Tensor, seqlens: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        start = 0
        for l in seqlens.tolist():
            end = start + int(l)
            out.append(packed[start:end])
            start = end
        return out

    def _apply_drop_path_per_seq(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        drop_module: nn.Module,
    ) -> torch.Tensor:
        if not self.training:
            return drop_module(x)
        if not isinstance(drop_module, DropPath):
            return drop_module(x)
        drop_prob = float(getattr(drop_module, "drop_prob", 0.0))
        if drop_prob <= 0.0:
            return x
        if x.ndim != 2:
            # Fallback for unexpected shapes.
            return drop_module(x)

        keep_prob = 1.0 - drop_prob
        if keep_prob <= 0.0:
            return x.new_zeros(x.shape)

        num_seq = int(seqlens.shape[0])
        gate = x.new_empty((num_seq, 1)).bernoulli_(keep_prob)
        if getattr(drop_module, "scale_by_keep", True):
            gate.div_(keep_prob)

        repeats = seqlens.to(device=x.device, dtype=torch.long)
        if int(repeats.sum().item()) != x.shape[0]:
            # Conservative fallback if sequence bookkeeping is inconsistent.
            return drop_module(x)
        gate_per_token = torch.repeat_interleave(gate, repeats, dim=0)  # [T, 1]
        return x * gate_per_token

    def _flash_attn_varlen_forward(
        self,
        attn_module: nn.Module,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "entropy_varlen_flash_enabled=True requires flash-attn. "
                "Install flash-attn in this environment."
            )
        if not x.is_cuda:
            raise RuntimeError(
                "entropy_varlen_flash_enabled=True requires a CUDA device. "
                "Current tensors are on CPU."
            )
        total_tokens, dim = x.shape
        num_heads = attn_module.num_heads
        head_dim = dim // num_heads

        qkv = attn_module.qkv(x).reshape(total_tokens, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(1)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=attn_module.attn_drop.p if self.training else 0.0,
            softmax_scale=getattr(attn_module, "scale", None),
            causal=False,
        )
        out = out.reshape(total_tokens, dim)
        out = attn_module.proj(out)
        out = attn_module.proj_drop(out)
        return out

    def _run_block_varlen(
        self,
        block: nn.Module,
        packed_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        residual = packed_tokens
        x_norm = block.norm1(packed_tokens)
        attn_out = self._flash_attn_varlen_forward(block.attn, x_norm, cu_seqlens, max_seqlen)
        if hasattr(block, "ls1"):
            attn_out = block.ls1(attn_out)
        if hasattr(block, "drop_path1"):
            attn_out = self._apply_drop_path_per_seq(attn_out, seqlens, block.drop_path1)
        elif hasattr(block, "drop_path"):
            attn_out = self._apply_drop_path_per_seq(attn_out, seqlens, block.drop_path)
        packed_tokens = residual + attn_out

        residual = packed_tokens
        x_mlp = block.mlp(block.norm2(packed_tokens))
        if hasattr(block, "ls2"):
            x_mlp = block.ls2(x_mlp)
        if hasattr(block, "drop_path2"):
            x_mlp = self._apply_drop_path_per_seq(x_mlp, seqlens, block.drop_path2)
        elif hasattr(block, "drop_path"):
            x_mlp = self._apply_drop_path_per_seq(x_mlp, seqlens, block.drop_path)
        packed_tokens = residual + x_mlp
        return packed_tokens

    def _project_mask_probs_with_source(
        self,
        mask_probs: torch.Tensor,
        source_vis: torch.Tensor,
    ) -> torch.Tensor:
        # mask_probs: [Q, N], source_vis: [M, N] -> [Q, M]
        if source_vis.numel() == 0:
            return mask_probs[:, :0]
        s = source_vis.float()
        denom = s.sum(dim=1).clamp(min=1.0)  # [M]
        merged = mask_probs @ s.transpose(0, 1)  # [Q, M]
        merged = merged / denom.unsqueeze(0)
        return merged

    def _project_mask_probs_with_global_history(
        self,
        mask_probs: torch.Tensor,
        merge_history: list[dict],
    ) -> torch.Tensor:
        if not merge_history:
            return mask_probs
        probs = mask_probs.transpose(0, 1).unsqueeze(0)  # [1, N, Q]
        for merge_info in merge_history:
            probs = self._merge_by_info(probs, merge_info, mode="mean")
        return probs.squeeze(0).transpose(0, 1)  # [Q, N_merged]

    def _unmerge_global_history_sample(
        self,
        seq: torch.Tensor,
        merge_history: list[dict],
    ) -> torch.Tensor:
        if not merge_history:
            return seq
        x = seq.unsqueeze(0)
        for merge_info in reversed(merge_history):
            x = self._unmerge_tokens(x, merge_info)
        return x.squeeze(0)

    def _unmerge_entropy_sample(
        self,
        seq: torch.Tensor,
        source_vis: torch.Tensor,
        has_queries: bool = True,
    ) -> torch.Tensor:
        protected = self.encoder.backbone.num_prefix_tokens + (self.num_q if has_queries else 0)
        if seq.shape[0] < protected:
            raise ValueError("Sequence shorter than protected tokens.")
        vision = seq[protected:]
        if source_vis.shape[0] != vision.shape[0]:
            raise ValueError(
                f"Entropy source/vision mismatch: {source_vis.shape[0]} vs {vision.shape[0]}"
            )
        if source_vis.shape[1] == 0:
            return seq
        idx = source_vis.argmax(dim=0)
        vision_full = vision.index_select(0, idx)
        return torch.cat((seq[:protected], vision_full), dim=0)

    def _unmerge_with_current_source_sample(
        self,
        seq: torch.Tensor,
        current_source: torch.Tensor,
        has_queries: bool = True,
    ) -> torch.Tensor:
        """
        Unmerge directly from current merged vision tokens to base tokens using a
        composed source map [N_current, N_base]. This avoids replaying global merge
        history + entropy unmerge for every routing step.
        """
        protected = self.encoder.backbone.num_prefix_tokens + (self.num_q if has_queries else 0)
        if seq.shape[0] < protected:
            raise ValueError("Sequence shorter than protected tokens.")
        vision = seq[protected:]
        if current_source.shape[0] != vision.shape[0]:
            raise ValueError(
                f"Current source/vision mismatch: {current_source.shape[0]} vs {vision.shape[0]}"
            )
        if current_source.shape[1] == 0:
            return seq
        idx = current_source.argmax(dim=0)
        vision_full = vision.index_select(0, idx)
        return torch.cat((seq[:protected], vision_full), dim=0)

    def _unmerge_with_base_to_current_sample(
        self,
        seq: torch.Tensor,
        base_to_current: torch.Tensor,
        has_queries: bool = True,
    ) -> torch.Tensor:
        protected = self.encoder.backbone.num_prefix_tokens + (self.num_q if has_queries else 0)
        if seq.shape[0] < protected:
            raise ValueError("Sequence shorter than protected tokens.")
        vision = seq[protected:]
        if base_to_current.numel() == 0:
            return seq
        if int(base_to_current.max().item()) >= vision.shape[0]:
            raise ValueError(
                f"base_to_current out of range: max={int(base_to_current.max().item())}, "
                f"vision_tokens={vision.shape[0]}"
            )
        vision_full = vision.index_select(0, base_to_current)
        return torch.cat((seq[:protected], vision_full), dim=0)

    def _project_mask_probs_with_base_to_current(
        self,
        mask_probs: torch.Tensor,
        base_to_current: torch.Tensor,
        n_current: int,
    ) -> torch.Tensor:
        # mask_probs: [Q, N_base], base_to_current: [N_base] -> [Q, N_current]
        if n_current <= 0:
            return mask_probs[:, :0]
        q, n_base = mask_probs.shape
        if base_to_current.numel() != n_base:
            raise ValueError(
                f"base_to_current length mismatch: {base_to_current.numel()} vs {n_base}"
            )
        out = mask_probs.new_zeros((q, n_current))
        idx = base_to_current.unsqueeze(0).expand(q, -1)
        out.scatter_add_(1, idx, mask_probs)
        counts = torch.bincount(base_to_current, minlength=n_current).to(mask_probs.dtype).clamp(min=1.0)
        out = out / counts.unsqueeze(0)
        return out

    def _build_old_to_new_index_bipartite(
        self,
        old_len: int,
        merge_info: dict,
        device: torch.device,
    ) -> torch.Tensor:
        if merge_info.get("type") != "bipartite":
            raise ValueError("Only bipartite merge info is supported for index-map update.")

        unm_idx = merge_info["unm_idx"].reshape(-1).to(device=device, dtype=torch.long)
        src_idx = merge_info["src_idx"].reshape(-1).to(device=device, dtype=torch.long)
        dst_idx = merge_info["dst_idx"].reshape(-1).to(device=device, dtype=torch.long)

        t1 = (old_len + 1) // 2  # even positions (src)
        t2 = old_len // 2        # odd positions (dst)
        r = src_idx.numel()

        old_to_new = torch.empty(old_len, device=device, dtype=torch.long)

        # Destination tokens (odd positions) are always present after merge.
        old_to_new[1::2] = (t1 - r) + torch.arange(t2, device=device, dtype=torch.long)

        # Unmerged source tokens map to the leading block.
        if unm_idx.numel() > 0:
            old_to_new[2 * unm_idx] = torch.arange(unm_idx.numel(), device=device, dtype=torch.long)

        # Merged source tokens map to their selected destination slots.
        if src_idx.numel() > 0:
            old_to_new[2 * src_idx] = (t1 - r) + dst_idx

        return old_to_new

    def _merge_tokens(
        self,
        x: torch.Tensor,
        layer_idx: int,
        merge_history: Optional[list] = None,
        merge_strategy: str = "project",
        local_merge_source: Optional[torch.Tensor] = None,
        entropy_keep_mask: Optional[torch.Tensor] = None,
    ):
        # Compute token merge masks. If we already merged, unmerge for mask prediction
        # but keep using the current merged tokens for the new merge.
        B, N, C = x.shape
        x_for_merge = x
        if merge_history or local_merge_source is not None:
            x_unmerged = x
            if merge_history:
                for merged_mask in reversed(merge_history):
                    x_unmerged = self._unmerge_tokens(x_unmerged, merged_mask)
            if local_merge_source is not None:
                x_unmerged = self._algm_unmerge(x_unmerged, local_merge_source, has_queries=True)
                B, N, C = x_unmerged.shape
            # mask_logits = self._merge_predict(self.encoder.backbone.norm(x_unmerged)) # [B, N, img_size // 4, img_size // 4]
            mask_logits, _ = self._predict(self.encoder.backbone.norm(x_unmerged)) # [B, N, img_size // 4, img_size // 4]
            # mask_logits = torch.randn(B, N, 128, 128, device=x.device)  # dummy logits for testing without ALGM
            if merge_strategy == "rematch":
                x_for_merge = x_unmerged
        else:
            mask_logits, _ = self._predict(self.encoder.backbone.norm(x)) # [B, N, img_size // 4, img_size // 4]
            # mask_logits = self._merge_predict(self.encoder.backbone.norm(x)) # [B, N, img_size // 4, img_size // 4]
            # mask_logits = torch.randn(B, N, 128, 128, device=x.device)  # dummy logits for testing without ALGM

        grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
        # interpolated = mask_logits
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

        projected_entropy_keep = None
        if entropy_keep_mask is not None:
            projected_entropy_keep = entropy_keep_mask
            if merge_history and merge_strategy == "project":
                projected_entropy_keep = self._project_token_values(projected_entropy_keep, merge_history)

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
        if projected_entropy_keep is not None and projected_entropy_keep.shape[-1] != vision_tokens.shape[1]:
            projected_entropy_keep = self._project_token_values_local(
                projected_entropy_keep, local_merge_source, has_queries=True
            )
            if not dynamo.is_compiling():
                if projected_entropy_keep.shape[-1] != vision_tokens.shape[1]:
                    warnings.warn(
                        f"Local entropy projection mismatch {projected_entropy_keep.shape[-1]} "
                        f"!= merged vision length {vision_tokens.shape[1]} at layer {layer_idx}."
                    )
                assert projected_entropy_keep.shape[-1] == vision_tokens.shape[1], (
                    f"Local entropy projection mismatch at layer {layer_idx}: "
                    f"{projected_entropy_keep.shape[-1]} vs {vision_tokens.shape[1]}"
                )

        if projected_entropy_keep is not None:
            entropy_keep = projected_entropy_keep.clamp(0.0, 1.0)
            entropy_merge = 1.0 - entropy_keep
            if self.entropy_tokenization_mode == "entropy_only":
                vision_token_masks = torch.stack(
                    (
                        entropy_keep * self.entropy_keep_channel_weight,
                        entropy_merge * self.entropy_merge_channel_weight,
                    ),
                    dim=1,
                )
            else:
                entropy_channel = (entropy_merge * self.entropy_merge_channel_weight).unsqueeze(1)
                vision_token_masks = torch.cat((vision_token_masks, entropy_channel), dim=1)
        merge, merge_info = global_bipartite_matching_masked(
            vision_tokens,
            vision_token_masks,
            layer_idx=layer_idx,
            tau=self.merge_conf_threshold,
            score_metric=self.global_merge_score_metric,
            min_merge_ratio=self.global_merge_min_ratio,
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
        # Keep this path allocation-free: infer whether query tokens are present
        # in source/x and adjust protected offsets instead of padding/cropping tensors.
        x_has_queries = bool(has_queries)
        source_has_queries = bool(has_queries)
        if source.shape[1] != x.shape[1]:
            delta = source.shape[1] - x.shape[1]
            if self.num_q > 0 and delta == self.num_q:
                source_has_queries = True
                x_has_queries = False
            elif self.num_q > 0 and delta == -self.num_q:
                source_has_queries = False
                x_has_queries = True
            else:
                raise ValueError(
                    "Source and x have different number of tokens. "
                    f"{source.shape[1]} != {x.shape[1]} "
                    f"(has_queries={has_queries}, num_q={self.num_q})"
                )

        source_protected = self.encoder.backbone.num_prefix_tokens + (
            self.num_q if source_has_queries else 0
        )
        x_protected = self.encoder.backbone.num_prefix_tokens + (
            self.num_q if x_has_queries else 0
        )

        src_vis = source[:, source_protected:, source_protected:]  # rows: merged vision, cols: original vision
        vision = x[:, x_protected:, :]

        if src_vis.shape[1] > vision.shape[1]:
            raise ValueError("Source has more merged tokens than x has vision tokens.")

        idxs = src_vis.argmax(dim=1)  # [B, original_vision_len]

        target_vision_len = (
            self.encoder.backbone.patch_embed.grid_size[0]
            * self.encoder.backbone.patch_embed.grid_size[1]
        )
        # if idxs.shape[1] != target_vision_len:
        #     if not dynamo.is_compiling():
        #         warnings.warn(
        #             "Number of original vision tokens does not match target. "
        #             f"{idxs.shape[1]} != {target_vision_len}. "
        #             "Using source-derived token count for unmerge."
        #         )

        x_ = torch.gather(
            vision,
            1,
            idxs.unsqueeze(-1).expand(-1, idxs.shape[1], vision.shape[2]),
        )

        x = torch.cat((x[:, :x_protected, :], x_), dim=1)

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
        entropy_maps: Optional[dict[int, torch.Tensor]] = None,
    ):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std
        # self._reset_turbo_info()
        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        # x, entropy_init_source, entropy_init_merged = self._entropy_pre_transformer_merge(
        #     x,
        #     entropy_keep_mask,
        # )

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
                and not self.entropy_pre_transformer_merge_enabled
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
                if local_merge_source is not None and self.num_q > 0:
                    if local_merge_source.shape[1] == x.shape[1] + self.num_q:
                        local_merge_source = local_merge_source[:, self.num_q :, :]
                        if local_merge_source.shape[2] >= self.num_q:
                            local_merge_source = local_merge_source[:, :, self.num_q :]
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
