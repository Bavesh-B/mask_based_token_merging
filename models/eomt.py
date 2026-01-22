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
from models.unmerge import LearnableUnmergeLayered


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        mask_logits_guided_token_merge=False,
        token_merge_layers: list=None,
        distill_layers: list = None,
        segloss_layers: list = None,
        merge_pool_size: int = 16,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(4))

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

        self.mask_logits_guided_token_merge = mask_logits_guided_token_merge
        if token_merge_layers is not None:
            self.token_merge_layers = sorted(token_merge_layers)
            assert self.token_merge_layers[0] >= len(self.encoder.backbone.blocks) - self.num_blocks
        else:
            self.token_merge_layers = [len(self.encoder.backbone.blocks) - self.num_blocks]

        self.distill_layers = distill_layers
        self.segloss_layers = sorted(segloss_layers) if segloss_layers is not None else []
        # self.unmerge_module = LearnableUnmergeLayered(dim=self.encoder.backbone.embed_dim)
        self.unmerge_module = None

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

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
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        ) # [B, N, 32, 32]
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        vision_token_masks = torch.softmax(interpolated, dim=1)

        # Merge tokens
        vision_tokens = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        mask = vision_token_masks.float()
        vision = vision_tokens.unsqueeze(1)
        masked_sum = (mask.unsqueeze(-1) * vision).sum(dim=2)
        counts = mask.sum(dim=2, keepdim=True)
        counts = counts.clamp(min=1e-6)
        fused = masked_sum / counts

        x = torch.cat(
            (x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], fused), dim=1
        )

        return x, mask

    def _unmerge_tokens(self, x: torch.Tensor, mask: torch.Tensor, layer_ratio: float = 1.0):
        # unmerge tokens
        if self.unmerge_module is None:
            fused = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
            fused_expanded = fused.unsqueeze(2)
            mask_expanded = mask.unsqueeze(-1)
            reproj_sum = (mask_expanded * fused_expanded).sum(dim=1)
            reproj_count = mask.sum(dim=1, keepdim=False).clamp(min=1e-6).unsqueeze(-1)  # [B, HW, 1]
            reprojected_tokens = reproj_sum / reproj_count
            x = torch.cat(
                (x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], reprojected_tokens), dim=1
            )
        else:
            fused = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
            x_unmerged = self.unmerge_module(fused, mask, layer_ratio)
            x = torch.cat(
                (x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], x_unmerged), dim=1
            )
        return x

    def forward(self, x: torch.Tensor, source: Optional[torch.Tensor] = None):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        distill_outputs = [] if self.distill_layers else None

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)
            
            ### seg loss outputs ###
            if ( i in self.segloss_layers ):
                if x.shape[1] == self.num_q + self.encoder.backbone.num_prefix_tokens + self.encoder.backbone.patch_embed.num_patches: 
                    mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                    mask_logits_per_layer.append(mask_logits)
                    class_logits_per_layer.append(class_logits)
                else:
                    unmerged = self._unmerge_tokens(x, mask, layer_ratio=i / len(self.encoder.backbone.blocks))
                    mask_logits, class_logits = self._predict(self.encoder.backbone.norm(unmerged))
                    mask_logits_per_layer.append(mask_logits)
                    class_logits_per_layer.append(class_logits)
            ### seg loss outputs ###

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

            ### token merge and unmerge ###
            if self.mask_logits_guided_token_merge:
                # Initial token merge
                if i == self.token_merge_layers[0]:
                    x, mask = self._merge_tokens(x)
                # Final unmerge
                elif i == len(self.encoder.backbone.blocks) - 1:
                    if x.shape[1] == self.num_q + self.num_q + self.encoder.backbone.num_prefix_tokens:
                        x = self._unmerge_tokens(x, mask, layer_ratio=i / len(self.encoder.backbone.blocks))
                    elif x.shape[1] == self.num_q + self.encoder.backbone.num_prefix_tokens + self.encoder.backbone.patch_embed.num_patches:
                        warnings.warn("Final unmerge skipped as number of tokens matches original.")
                    else:
                        raise ValueError("Unexpected number of tokens during final unmerge.")
            ### token merge and unmerge ###

            ### distillation outputs ###
            if distill_outputs is not None and i in self.distill_layers:
                if x.shape[1] == self.num_q + self.encoder.backbone.num_prefix_tokens + self.encoder.backbone.patch_embed.num_patches:
                    vision_tokens = x[
                        :, self.num_q + self.encoder.backbone.num_prefix_tokens :, :
                    ]
                    distill_outputs.append(vision_tokens)
                else:
                    unmerged = self._unmerge_tokens(x, mask, layer_ratio=i / len(self.encoder.backbone.blocks))
                    vision_tokens = unmerged[
                        :, self.num_q + self.encoder.backbone.num_prefix_tokens :, :
                    ]
                    distill_outputs.append(vision_tokens)
            ### distillation outputs ###
                
        #### This is code for ALGM token merging baseline. ######
        if source is not None:
            idxs = source[:, self.num_q + self.encoder.backbone.num_prefix_tokens:, self.num_q + self.encoder.backbone.num_prefix_tokens:].argmax(dim=1)
            # defining a torch tensor with the output size 
            masks_shape = x.shape
            size_x = source.shape[2]-1
            x_ =  torch.ones(masks_shape[0],size_x,masks_shape[2],device=x.device)
        
            for batch in range(0,masks_shape[0]):
                x_[batch,:,:] = x[batch,idxs[batch]]
            
            x = torch.cat((x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :], x_), dim=1)
        #### End of ALGM token merging baseline. ######

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
            distill_outputs,
        )
