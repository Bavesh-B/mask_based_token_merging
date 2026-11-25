# ---------------------------------------------------------------
# mask2former/modeling/backbone/dinov2_cts.py
#
# DINOv2 ViT-L/16 backbone with CTS / ToMe token reduction
# for facebookresearch/Mask2Former (Detectron2).
#
# Drop this file into:
#   mask2former/modeling/backbone/dinov2_cts.py
#
# Then add to mask2former/modeling/backbone/__init__.py:
#   from .dinov2_cts import D2DINOv2WithCTS
#
# And add to mask2former/modeling/__init__.py:
#   from .backbone.dinov2_cts import D2DINOv2WithCTS
#
# CTS based on:
#   Lu et al., "Content-aware Token Sharing for Efficient Semantic
#   Segmentation with Vision Transformers", CVPR 2023
#   https://github.com/tue-mps/cts-segmenter
#
# CTS/ToMe logic ported verbatim from EoMT
# (Mobile Perception Systems Lab, TU/e), MIT License.
#
# ---------------------------------------------------------------
#
# TOKEN COUNT REFERENCE  (patch 16, matched to EoMT exactly)
# ===========================================================
# ADE20k / COCO  — 640×640  → 40×40 = 1600 spatial tokens
#   ToMe:  r=750  → 850 tokens retained  (53.1%)
#   CTS:   notshared=600, shared=250  → 850 tokens  ✓
#
# Cityscapes     — 1024×1024 → 64×64 = 4096 spatial tokens
#   ToMe:  r=1095 → 3001 tokens retained (73.3%)
#   CTS:   notshared=2636, shared=365  → 3001 tokens ✓
#   Derivation: (1024-365)*4 + 365 = 3001
#
# REPO INTEGRATION
# ================
# The pixel decoder (MSDeformAttnPixelDecoder) expects features keyed
# as "res2", "res3", "res4", "res5" with strides 4, 8, 16, 32.
# We output a single feature map at stride 16 and derive the other
# three via lightweight projection + interpolation so the rest of the
# Mask2Former pipeline requires zero changes.
# ---------------------------------------------------------------

from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec
from detectron2.modeling.backbone import Backbone


# ============================================================================
# CTS CONFIGURATION  (verbatim from EoMT)
# ============================================================================
class CTSConfig:
    ENABLE_CTS   = True
    CTS_IN_TRAIN = True
    CTS_IN_EVAL  = True

    POLICY_NET_PATH = "../eomt_cts/policynet/logdir/policynet_efficientnet_ade20k/model.pth"   # set via cfg.MODEL.DINOV2_CTS.POLICY_NET_PATH

    # ADE20k / COCO  (40×40 patch grid → 20×20 superpatch grid)
    NUM_TOKENS_NOTSHARED_ADE20K = 600
    NUM_TOKENS_SHARED_ADE20K    = 250   # total 850 == 1600 - 750  ✓

    # Cityscapes     (64×64 patch grid → 32×32 superpatch grid)
    NUM_TOKENS_NOTSHARED_CITYSCAPES = 2636
    NUM_TOKENS_SHARED_CITYSCAPES    = 365  # total 3001 == 4096 - 1095  ✓

    DEBUG_PRINT_SHAPES = False
    DEBUG_CHECK_NAN    = False


# ============================================================================
# TOME CONFIGURATION  (verbatim from EoMT)
# ============================================================================
class ToMeConfig:
    ENABLE_TOME   = False
    TOME_IN_TRAIN = False
    TOME_IN_EVAL  = False

    # Verbatim from EoMT — layer 8 preserved (DINOv2 ViT-L also has 24 blocks)
    TOME_MERGE_SCHEDULE_ADE20K     = [(8, 750)]
    TOME_MERGE_SCHEDULE_CITYSCAPES = [(8, 1095)]
    TOME_MERGE_SCHEDULE_COCO       = [(8, 750)]

    # Unmerge happens after block 23, before FPN (replaces EoMT's split_idx=20)
    TOME_UNMERGE_BEFORE_FPN = True


# ============================================================================
# TOKEN SHARING  (verbatim from EoMT)
# ============================================================================
class TokenSharing(nn.Module):
    """Content-aware token sharing. Verbatim from EoMT / cts-segmenter."""

    def __init__(
        self,
        num_tokens_notshared: int,
        num_tokens_shared: int,
        grid_size: Tuple[int, int],
    ):
        super().__init__()
        self.num_tokens_notshared = num_tokens_notshared
        self.num_tokens_shared    = num_tokens_shared
        self.grid_size            = grid_size
        self.sp_grid_size         = (grid_size[0] // 2, grid_size[1] // 2)
        self.total_superpatches   = self.sp_grid_size[0] * self.sp_grid_size[1]

    def select_superpatches_to_share(
        self, policy_scores: torch.Tensor
    ) -> torch.Tensor:
        B = policy_scores.shape[0]
        scores_flat = policy_scores.view(B, -1)
        _, indices  = torch.topk(
            scores_flat, k=self.num_tokens_shared, dim=1, sorted=False
        )
        return indices

    def share_tokens(
        self,
        patch_tokens: torch.Tensor,
        superpatch_indices: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        H, W    = self.grid_size

        tokens_spatial = patch_tokens.view(B, H, W, C)
        tokens_sp = tokens_spatial.view(
            B, self.sp_grid_size[0], 2, self.sp_grid_size[1], 2, C
        )
        tokens_sp = tokens_sp.permute(0, 1, 3, 2, 4, 5).contiguous()
        tokens_sp = tokens_sp.view(B, -1, 2, 2, C)
        tokens_sp_pooled = tokens_sp.mean(dim=[2, 3])

        sp_mask = torch.zeros(
            B, self.total_superpatches, dtype=torch.bool, device=patch_tokens.device
        )
        sp_mask.scatter_(1, superpatch_indices, True)

        num_shared       = superpatch_indices.shape[1]
        num_notshared_sp = self.total_superpatches - num_shared

        shared    = tokens_sp_pooled[sp_mask].view(B, num_shared, C)
        notshared = tokens_sp[~sp_mask].view(B, num_notshared_sp, 4, C).reshape(B, -1, C)

        return torch.cat([notshared, shared], dim=1)

    @torch.compiler.disable
    def unshare_tokens(
        self,
        shared_tokens: torch.Tensor,
        superpatch_indices: torch.Tensor,
    ) -> torch.Tensor:
        B, M, C = shared_tokens.shape
        H, W    = self.grid_size
        N       = H * W

        num_shared    = superpatch_indices.shape[1]
        if num_shared == 0:
            return shared_tokens

        num_notshared = shared_tokens.shape[1] - num_shared
        notshared     = shared_tokens[:, :num_notshared, :]
        shared        = shared_tokens[:, num_notshared:, :]

        sp_mask = torch.zeros(
            B, self.total_superpatches, dtype=torch.bool, device=shared_tokens.device
        )
        sp_mask.scatter_(1, superpatch_indices, True)

        num_notshared_sp = self.total_superpatches - num_shared
        notshared = notshared.view(B, num_notshared_sp, 4, C)
        shared    = shared.unsqueeze(2).expand(B, num_shared, 4, C)

        output_sp = torch.zeros(
            B, self.total_superpatches, 4, C,
            device=shared_tokens.device, dtype=shared_tokens.dtype,
        )
        output_sp[sp_mask]  = shared.reshape(-1, 4, C)
        output_sp[~sp_mask] = notshared.reshape(-1, 4, C)

        output_sp_grid  = output_sp.view(
            B, self.sp_grid_size[0], self.sp_grid_size[1], 2, 2, C
        )
        output_sp_grid  = output_sp_grid.permute(0, 1, 3, 2, 4, 5)
        output_patches  = output_sp_grid.contiguous().view(B, H, W, C)
        return output_patches.view(B, N, C)


# ============================================================================
# POLICY NETWORK LOADER  (verbatim from EoMT)
# ============================================================================
def load_policy_network(checkpoint_path: str) -> Optional[nn.Module]:
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Policy network not found at {checkpoint_path}. CTS disabled.")
        return None
    try:
        print(f"Loading policy network from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            policy_net = checkpoint
        elif isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                policy_net = checkpoint["model"]
            else:
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                if isinstance(state_dict, dict) and all(
                    isinstance(v, torch.Tensor) for v in state_dict.values()
                ):
                    try:
                        from policynet.net import PolicyNet
                    except ModuleNotFoundError:
                        import sys
                        policynet_root = Path(__file__).resolve().parents[3] / "policynet"
                        sys.path.insert(0, str(policynet_root))
                        try:
                            from policynet.net import PolicyNet
                        except ModuleNotFoundError as e:
                            print(f"ERROR: Failed to import PolicyNet: {e}")
                            return None
                    try:
                        policy_net = PolicyNet()
                        policy_net.load_state_dict(state_dict, strict=True)
                    except Exception as e:
                        print(f"ERROR: Failed to load PolicyNet state dict: {e}")
                        return None
                else:
                    print("WARNING: Checkpoint format not recognized. CTS disabled.")
                    return None
        else:
            print(f"ERROR: Unknown checkpoint format: {type(checkpoint)}")
            return None

        policy_net.eval()
        for p in policy_net.parameters():
            p.requires_grad = False

        print(f"Policy network loaded: {type(policy_net)}")
        return policy_net

    except Exception as e:
        import traceback
        print(f"ERROR loading policy network: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# TOME REDUCER  (verbatim from EoMT / tome_reducer_scatter.py)
# ============================================================================
class ToMeReducer(nn.Module):
    """Token Merging via bipartite soft matching. Verbatim from EoMT."""

    def __init__(self):
        super().__init__()

    @torch.compiler.disable
    def merge_tokens(
        self, tokens: torch.Tensor, r: int
    ) -> tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        B, N, C = tokens.shape
        if r <= 0 or N < 2:
            return tokens, None

        a, b   = tokens[:, 0::2, :], tokens[:, 1::2, :]
        n_a, n_b = a.shape[1], b.shape[1]
        if n_b == 0:
            return tokens, None
        r = min(r, n_b)
        if r <= 0:
            return tokens, None

        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        sim    = a_norm @ b_norm.transpose(-2, -1)

        node_max        = sim.max(dim=1)[0]
        _, dst_idx      = torch.topk(node_max, k=r, dim=1)
        src_idx         = sim.max(dim=1)[1]
        batch_idx       = torch.arange(B, device=tokens.device).unsqueeze(1)
        src_idx         = src_idx[batch_idx, dst_idx]

        merged = (a[batch_idx, src_idx] + b[batch_idx, dst_idx]) / 2

        a_keep = torch.ones(B, n_a, dtype=torch.bool, device=tokens.device)
        a_keep.scatter_(1, src_idx, False)
        b_keep = torch.ones(B, n_b, dtype=torch.bool, device=tokens.device)
        b_keep.scatter_(1, dst_idx, False)

        a_keep_idx = torch.argsort(a_keep.int(), dim=1, descending=True)[:, : n_a - r]
        b_keep_idx = torch.argsort(b_keep.int(), dim=1, descending=True)[:, : n_b - r]

        unmerged_a = torch.gather(a, 1, a_keep_idx.unsqueeze(-1).expand(-1, -1, C))
        unmerged_b = torch.gather(b, 1, b_keep_idx.unsqueeze(-1).expand(-1, -1, C))

        state = {
            "N": N, "r": r, "n_a": n_a, "n_b": n_b,
            "src_idx": src_idx, "dst_idx": dst_idx,
        }
        return torch.cat([unmerged_a, unmerged_b, merged], dim=1), state

    @torch.compiler.disable
    def unmerge_tokens(
        self, merged_tokens: torch.Tensor, state: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        if state is None:
            return merged_tokens

        B, _, C  = merged_tokens.shape
        N, r, n_a, n_b = state["N"], state["r"], state["n_a"], state["n_b"]
        src_idx, dst_idx = state["src_idx"], state["dst_idx"]

        n_ua = n_a - r
        n_ub = n_b - r
        unmerged_a = merged_tokens[:, :n_ua, :]
        unmerged_b = merged_tokens[:, n_ua : n_ua + n_ub, :]
        merged     = merged_tokens[:, n_ua + n_ub :, :]

        a_rec = torch.zeros(B, n_a, C, device=merged_tokens.device, dtype=merged_tokens.dtype)
        b_rec = torch.zeros(B, n_b, C, device=merged_tokens.device, dtype=merged_tokens.dtype)

        batch_idx = torch.arange(B, device=merged_tokens.device).unsqueeze(1)
        a_rec[batch_idx, src_idx] = merged
        b_rec[batch_idx, dst_idx] = merged

        a_merged_mask = torch.zeros(B, n_a, dtype=torch.bool, device=merged_tokens.device)
        a_merged_mask.scatter_(1, src_idx, True)
        b_merged_mask = torch.zeros(B, n_b, dtype=torch.bool, device=merged_tokens.device)
        b_merged_mask.scatter_(1, dst_idx, True)

        a_pos = (~a_merged_mask).int().argsort(dim=1, descending=True)[:, :n_ua]
        b_pos = (~b_merged_mask).int().argsort(dim=1, descending=True)[:, :n_ub]

        a_rec.scatter_(1, a_pos.unsqueeze(-1).expand(-1, -1, C), unmerged_a)
        b_rec.scatter_(1, b_pos.unsqueeze(-1).expand(-1, -1, C), unmerged_b)

        output = torch.zeros(B, N, C, device=merged_tokens.device, dtype=merged_tokens.dtype)
        output[:, 0::2, :] = a_rec
        output[:, 1::2, :] = b_rec
        return output


# ============================================================================
# MULTI-SCALE PROJECTION
# Produces res2/res3/res4/res5 from a single ViT feature map so the
# MSDeformAttnPixelDecoder requires zero changes.
# ============================================================================
class MultiScaleProjection(nn.Module):
    """
    Projects a single (B, C, H, W) ViT feature map at stride 16 to the
    four scales Mask2Former's pixel decoder expects:

        res2  stride 4   (upsample ×4)
        res3  stride 8   (upsample ×2)
        res4  stride 16  (identity)
        res5  stride 32  (downsample ×2)

    Each scale gets its own 1×1 conv to give the pixel decoder independent
    channels to work with, matching what a Swin FPN would produce.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.proj_res2 = nn.Conv2d(in_channels, out_channels, 1)
        self.proj_res3 = nn.Conv2d(in_channels, out_channels, 1)
        self.proj_res4 = nn.Conv2d(in_channels, out_channels, 1)
        self.proj_res5 = nn.Conv2d(in_channels, out_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W) at stride 16
        Returns:
            dict with keys res2, res3, res4, res5
        """
        res4 = self.proj_res4(feat)
        res3 = self.proj_res3(F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False))
        res2 = self.proj_res2(F.interpolate(feat, scale_factor=4.0, mode="bilinear", align_corners=False))
        res5 = self.proj_res5(F.avg_pool2d(feat, kernel_size=2, stride=2))
        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}


# ============================================================================
# MAIN BACKBONE
# ============================================================================
@BACKBONE_REGISTRY.register()
class D2DINOv2WithCTS(Backbone):
    """
    DINOv2 ViT-L/16 backbone with CTS / ToMe token reduction,
    registered for facebookresearch/Mask2Former (Detectron2).

    Drop-in replacement for D2SwinTransformer.  The rest of the
    Mask2Former pipeline (pixel decoder, transformer decoder, losses)
    requires no changes.

    Config keys (add to your yaml under MODEL.DINOV2_CTS):
        BACKBONE_NAME   str    timm model name
        PATCH_SIZE      int    16
        IMG_SIZE        int    640 or 1024
        DATASET         str    "ade20k" | "cityscapes" | "coco"
        OUT_CHANNELS    int    256  (proj channels fed to pixel decoder)
        ENABLE_CTS      bool
        CTS_IN_TRAIN    bool
        CTS_IN_EVAL     bool
        POLICY_NET_PATH str
        ENABLE_TOME     bool
        TOME_IN_TRAIN   bool
        TOME_IN_EVAL    bool

    Structural differences vs EoMT (all others are verbatim):
        - No query token injection in the block loop.
        - No masked attention (_attn_mask / _predict).
        - Unshare/unmerge after block 23, before multi-scale projection,
          instead of at EoMT's split_idx=20.
        - Returns {"res2", "res3", "res4", "res5"} feature dict instead
          of (mask_logits, class_logits).
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # ── config ──────────────────────────────────────────────────────
        dcfg = cfg.MODEL.DINOV2_CTS

        backbone_name = dcfg.BACKBONE_NAME   # e.g. "vit_large_patch14_dinov2"
        patch_size    = dcfg.PATCH_SIZE       # 16
        img_size      = dcfg.IMG_SIZE         # 640 or 1024
        dataset       = dcfg.DATASET.lower()  # "ade20k" | "cityscapes" | "coco"
        out_channels  = dcfg.OUT_CHANNELS     # 256

        # ── timm ViT ────────────────────────────────────────────────────
        self.vit = timm.create_model(
            backbone_name,
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
            dynamic_img_size=True
        )
        self.num_prefix_tokens = self.vit.num_prefix_tokens
        self.grid_size         = self.vit.patch_embed.grid_size  # (H, W)
        self.embed_dim         = self.vit.embed_dim

        # ── CTS ─────────────────────────────────────────────────────────
        self.enable_cts   = dcfg.ENABLE_CTS
        self.cts_in_train = dcfg.CTS_IN_TRAIN
        self.cts_in_eval  = dcfg.CTS_IN_EVAL

        if dataset == "cityscapes":
            num_notshared = CTSConfig.NUM_TOKENS_NOTSHARED_CITYSCAPES
            num_shared    = CTSConfig.NUM_TOKENS_SHARED_CITYSCAPES
        else:
            num_notshared = CTSConfig.NUM_TOKENS_NOTSHARED_ADE20K
            num_shared    = CTSConfig.NUM_TOKENS_SHARED_ADE20K

        if self.enable_cts:
            self.token_sharing = TokenSharing(
                num_tokens_notshared=num_notshared,
                num_tokens_shared=num_shared,
                grid_size=self.grid_size,
            )
            self.policy_net = load_policy_network(dcfg.POLICY_NET_PATH)
            if self.policy_net is None:
                self.enable_cts    = False
                self.token_sharing = None
        else:
            self.token_sharing = None
            self.policy_net    = None

        self._policy_net_device = None

        # ── ToMe ────────────────────────────────────────────────────────
        self.enable_tome   = dcfg.ENABLE_TOME
        self.tome_in_train = dcfg.TOME_IN_TRAIN
        self.tome_in_eval  = dcfg.TOME_IN_EVAL

        if dataset == "cityscapes":
            self.tome_merge_schedule = ToMeConfig.TOME_MERGE_SCHEDULE_CITYSCAPES
        elif dataset == "coco":
            self.tome_merge_schedule = ToMeConfig.TOME_MERGE_SCHEDULE_COCO
        else:
            self.tome_merge_schedule = ToMeConfig.TOME_MERGE_SCHEDULE_ADE20K

        self.tome_reducer = ToMeReducer() if self.enable_tome else None

        # ── multi-scale projection → pixel decoder ───────────────────────
        self.ms_proj = MultiScaleProjection(
            in_channels=self.embed_dim,
            out_channels=out_channels,
        )
        self._out_channels = out_channels

        # ── summary ─────────────────────────────────────────────────────
        H, W  = self.grid_size
        total = H * W
        print("=" * 70)
        print(f"D2DINOv2WithCTS  [{dataset}]  {img_size}×{img_size}  patch {patch_size}")
        print(f"  Grid: {H}×{W}  spatial tokens: {total}  prefix: {self.num_prefix_tokens}")
        print(f"  CTS:  enabled={self.enable_cts}  train={self.cts_in_train}  eval={self.cts_in_eval}")
        if self.enable_cts:
            print(f"    notshared={num_notshared}  shared={num_shared}  → {num_notshared+num_shared} tokens")
        print(f"  ToMe: enabled={self.enable_tome}  train={self.tome_in_train}  eval={self.tome_in_eval}")
        if self.enable_tome:
            cur = total
            for layer, r in self.tome_merge_schedule:
                cur -= r
                print(f"    layer {layer}: merge {r} → {cur} tokens ({cur/total*100:.1f}% retained)")
        print("=" * 70)

    # ── Detectron2 Backbone interface ────────────────────────────────────────

    def output_shape(self) -> Dict[str, ShapeSpec]:
        c = self._out_channels
        return {
            "res2": ShapeSpec(channels=c, stride=4),
            "res3": ShapeSpec(channels=c, stride=8),
            "res4": ShapeSpec(channels=c, stride=16),
            "res5": ShapeSpec(channels=c, stride=32),
        }

    # ── Internal helpers (verbatim from EoMT) ───────────────────────────────

    def _get_rope(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if hasattr(self.vit, "rope_embeddings"):
            return self.vit.rope_embeddings(x)
        return None

    def _attn(
        self,
        block: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Verbatim from EoMT._attn()."""
        module = block.attn if hasattr(block, "attn") else block.attention

        if rope is not None:
            if mask is not None:
                mask = mask[:, None].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0
        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x   = attn @ v

        return module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

    def _block_forward(
        self,
        block: nn.Module,
        x: torch.Tensor,
        rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Verbatim from EoMT block forward."""
        attn_out = self._attn(block, block.norm1(x), mask=None, rope=rope)

        if hasattr(block, "ls1"):
            x = x + block.ls1(attn_out)
        elif hasattr(block, "layer_scale1"):
            x = x + block.layer_scale1(attn_out)
        else:
            x = x + attn_out

        mlp_out = block.mlp(block.norm2(x))

        if hasattr(block, "ls2"):
            x = x + block.ls2(mlp_out)
        elif hasattr(block, "layer_scale2"):
            x = x + block.layer_scale2(mlp_out)
        else:
            x = x + mlp_out

        return x

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) — normalised by Mask2Former's preprocess step.

        Returns:
            dict {"res2", "res3", "res4", "res5"} of (B, C, H', W') tensors
            at strides 4, 8, 16, 32 respectively.

        Forward flow (matching EoMT exactly):
            1. Determine CTS / ToMe active flags (same mutual-exclusion guard).
            2. Policy network → superpatch indices (CTS).
            3. Patch embed.
            4. Pos embed then CTS share (applied to spatial tokens only,
               prefix tokens untouched) — verbatim from EoMT.
            5. Block loop: ToMe merge at scheduled layers, block forward.
               No query injection, no masked attention.
            6. After block 23: CTS unshare or ToMe unmerge — this replaces
               EoMT's unshare-at-split_idx because there is no decoder split
               inside the backbone.
            7. Norm, reshape to (B, C, H, W), multi-scale projection.
        """

        # ── 1. active flags (verbatim from EoMT) ────────────────────────
        cts_active = self.enable_cts and (
            (self.training and self.cts_in_train)
            or (not self.training and self.cts_in_eval)
        )
        tome_active = self.enable_tome and (
            (self.training and self.tome_in_train)
            or (not self.training and self.tome_in_eval)
        )
        # mutual exclusion — verbatim from EoMT
        if cts_active and tome_active:
            tome_active = False

        # ── 2. policy network (verbatim from EoMT) ───────────────────────
        superpatch_indices = None
        if cts_active and self.policy_net is not None:
            with torch.no_grad():
                try:
                    if self._policy_net_device != x.device:
                        self.policy_net = self.policy_net.to(x.device)
                        self._policy_net_device = x.device
                    policy_scores = self.policy_net(x)
                    if isinstance(policy_scores, dict):
                        policy_scores = policy_scores.get("logits", policy_scores)
                    if policy_scores.dim() == 4 and policy_scores.shape[1] == 2:
                        policy_scores = policy_scores[:, 1:2]
                    superpatch_indices = self.token_sharing.select_superpatches_to_share(
                        policy_scores
                    )
                except Exception as e:
                    print(f"WARNING: Policy network failed: {e}. Disabling CTS for batch.")
                    superpatch_indices = None

        cts_reduced  = cts_active and superpatch_indices is not None
        tome_reduced = False
        tome_states: List[Dict[str, Any]] = []

        # ── 3. patch embed ───────────────────────────────────────────────
        tokens = self.vit.patch_embed(x)
        rope   = self._get_rope(x)

        # ── 4. pos embed + CTS share (verbatim from EoMT) ────────────────
        if hasattr(self.vit, "_pos_embed"):
            if cts_reduced:
                x_full    = self.vit._pos_embed(tokens)
                x_prefix  = x_full[:, : self.num_prefix_tokens, :]
                x_spatial = x_full[:, self.num_prefix_tokens:, :]
                x_spatial = self.token_sharing.share_tokens(x_spatial, superpatch_indices)
                tokens    = torch.cat([x_prefix, x_spatial], dim=1)
            else:
                tokens = self.vit._pos_embed(tokens)

        # ── 5. block loop (verbatim from EoMT, minus query + masked attn) ─
        for i, block in enumerate(self.vit.blocks):

            # ToMe progressive merge at scheduled layers
            if tome_active and self.tome_reducer is not None:
                for merge_layer, r_to_merge in self.tome_merge_schedule:
                    if i == merge_layer:
                        x_sp = tokens[:, self.num_prefix_tokens:, :]
                        x_sp, state = self.tome_reducer.merge_tokens(x_sp, int(r_to_merge))
                        if state is not None:
                            tome_states.append(state)
                            tome_reduced = True
                            tokens = torch.cat(
                                [tokens[:, : self.num_prefix_tokens, :], x_sp], dim=1
                            )
                        break

            tokens = self._block_forward(block, tokens, rope)

        # ── 6. unshare / unmerge after last block (replaces split_idx) ───
        if cts_reduced:
            x_sp   = tokens[:, self.num_prefix_tokens:, :]
            x_sp   = self.token_sharing.unshare_tokens(x_sp, superpatch_indices)
            tokens = torch.cat([tokens[:, : self.num_prefix_tokens, :], x_sp], dim=1)

        if tome_reduced and self.tome_reducer is not None:
            x_sp = tokens[:, self.num_prefix_tokens:, :]
            for state in reversed(tome_states):
                x_sp = self.tome_reducer.unmerge_tokens(x_sp, state)
            tokens = torch.cat([tokens[:, : self.num_prefix_tokens, :], x_sp], dim=1)

        # ── 7. norm → (B, C, H, W) → multi-scale projection ─────────────
        # H, W    = self.grid_size
        # spatial = tokens[:, self.num_prefix_tokens:, :]       # strip prefix
        # spatial = self.vit.norm(spatial)                       # final LN
        # feat    = spatial.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        spatial = tokens[:, self.num_prefix_tokens:, :]       # strip prefix
        spatial = self.vit.norm(spatial)                       # final LN
        # Compute actual grid size dynamically (dynamic_img_size=True means
        # H*W varies per batch depending on input resolution)
        B, N, C = spatial.shape
        H = x.shape[2] // self.vit.patch_embed.patch_size[0]
        W = x.shape[3] // self.vit.patch_embed.patch_size[1]
        feat    = spatial.transpose(1, 2).reshape(B, -1, H, W)
        return self.ms_proj(feat)   # {"res2", "res3", "res4", "res5"}