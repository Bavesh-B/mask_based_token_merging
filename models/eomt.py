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

from models.scale_block import ScaleBlock
from algm_addition import apply_patch
from einops import rearrange


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        merge_pool_size: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.merge_pool_size = merge_pool_size

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

        # print('SAHPEEE!', x.shape, self.encoder.backbone.num_prefix_tokens)
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

    def _merge_tokens(self, x: torch.Tensor):
        # Compute token merge masks
        mask_logits, _ = self._predict(self.encoder.backbone.norm(x)) # [B, N, 128, 128]
        grid_h, grid_w = self.encoder.backbone.patch_embed.grid_size
        interpolated = F.interpolate(
            mask_logits,
            (grid_h, grid_w),
            mode="bilinear",
        ) # [B, N, 32, 32]
        vision_token_masks = torch.softmax(interpolated, dim=1)

        # Merge tokens with spatial pooling to keep multiple tokens per mask.
        pool_size = max(1, int(self.merge_pool_size))
        pad_h = (pool_size - (grid_h % pool_size)) % pool_size
        pad_w = (pool_size - (grid_w % pool_size)) % pool_size
        if pad_h or pad_w:
            vision_token_masks = F.pad(vision_token_masks, (0, pad_w, 0, pad_h))

        vision_tokens = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        vision = vision_tokens.transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)
        if pad_h or pad_w:
            vision = F.pad(vision, (0, pad_w, 0, pad_h))

        weighted = vision_token_masks.unsqueeze(2) * vision.unsqueeze(1)
        weighted = weighted.flatten(0, 1)
        pooled_weighted = F.avg_pool2d(weighted, pool_size, stride=pool_size)
        pooled_weighted = pooled_weighted * (pool_size * pool_size)
        pooled_weighted = pooled_weighted.view(
            x.shape[0],
            vision_token_masks.shape[1],
            vision.shape[1],
            pooled_weighted.shape[-2],
            pooled_weighted.shape[-1],
        )

        pooled_mask = F.avg_pool2d(vision_token_masks, pool_size, stride=pool_size)
        pooled_mask = pooled_mask * (pool_size * pool_size)
        pooled_mask = pooled_mask.clamp(min=1e-6)

        fused = pooled_weighted / pooled_mask.unsqueeze(2)
        fused = fused.permute(0, 1, 3, 4, 2).reshape(x.shape[0], -1, vision.shape[1])

        x = torch.cat(
            (x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], fused), dim=1
        )

        mask_info = {
            "weights": vision_token_masks,
            "pool_size": pool_size,
            "pad_hw": (pad_h, pad_w),
            "grid_hw": (grid_h, grid_w),
        }

        return x, mask_info

    def _unmerge_tokens(self, x: torch.Tensor, mask):
        # unmerge tokens
        if isinstance(mask, dict):
            weights = mask["weights"]
            pool_size = mask["pool_size"]
            pad_h, pad_w = mask["pad_hw"]
            grid_h, grid_w = mask["grid_hw"]

            fused = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
            bsz = fused.shape[0]
            num_masks = weights.shape[1]
            h_pad, w_pad = weights.shape[-2], weights.shape[-1]
            h_tiles = h_pad // pool_size
            w_tiles = w_pad // pool_size

            fused = fused.view(bsz, num_masks, h_tiles, w_tiles, -1)
            fused = fused.permute(0, 1, 4, 2, 3)
            fused_up = F.interpolate(fused, size=(h_pad, w_pad), mode="nearest")

            pooled_mask = F.avg_pool2d(weights, pool_size, stride=pool_size)
            pooled_mask = pooled_mask * (pool_size * pool_size)
            pooled_mask = pooled_mask.clamp(min=1e-6)
            pooled_mask_up = F.interpolate(pooled_mask, size=(h_pad, w_pad), mode="nearest")
            norm_weights = weights / pooled_mask_up

            reprojected = (fused_up * norm_weights.unsqueeze(2)).sum(dim=1)
            if pad_h or pad_w:
                reprojected = reprojected[..., :grid_h, :grid_w]
            reprojected_tokens = reprojected.flatten(2).transpose(1, 2)
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

        self._reset_turbo_info()
        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        mask_logits_per_layer, class_logits_per_layer = [], []
        turbo_info = getattr(self.encoder.backbone, "_turbo_info", None)
        num_protected_tokens = self.encoder.backbone.num_prefix_tokens
        source = turbo_info["source"] if turbo_info is not None else None

        for i, block in enumerate(self.encoder.backbone.blocks):
            # if i>=8:
            #     assert False, 'Local merge should be done after 1'
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
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
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if hasattr(block, "_turbo_info"):
                # print('i:', i)
                x = block(x, num_protected_tokens=num_protected_tokens)
                # source = self.encoder.backbone._turbo_info["source"]
            else:
                x = block(x)
        # source = turbo_info["source"]
        #### This is code for ALGM token merging baseline. ######
        if self.encoder.backbone._turbo_info["source"] is not None:
            x_vis = x[:, self.num_q:, :]
            x_vis = self._algm_unmerge(x_vis, self.encoder.backbone._turbo_info["source"], has_queries=False)
            x = torch.cat((x[:, :self.num_q, :], x_vis), dim=1)

        #### End of ALGM token merging baseline. ######

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
