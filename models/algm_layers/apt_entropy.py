from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def compute_patch_entropy_batched(
    images: torch.Tensor,
    patch_size: int = 16,
    num_scales: int = 2,
    bins: int = 64,
    pad_value: float = 1e6,
) -> Dict[int, torch.Tensor]:
    """
    APT-style batched patch entropy maps for multiple scales.
    Args:
        images: [B, C, H, W], expected in [0, 255].
    Returns:
        Dict[patch_size, entropy_map], each entropy map is [B, H_ps, W_ps].
    """
    if images.dim() != 4:
        raise ValueError(f"images must be [B, C, H, W], got {images.shape}")
    if bins <= 1:
        raise ValueError(f"bins must be > 1, got {bins}")

    bsz, channels, height, width = images.shape
    device = images.device

    if channels == 3:
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        grayscale = (images * weights).sum(dim=1)
    else:
        grayscale = images[:, 0]

    entropy_maps: Dict[int, torch.Tensor] = {}
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]

    for ps in patch_sizes:
        num_h = (height + ps - 1) // ps
        num_w = (width + ps - 1) // ps
        pad_h = num_h * ps - height
        pad_w = num_w * ps - width

        padded = F.pad(grayscale, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        patches = padded.unfold(1, ps, ps).unfold(2, ps, ps)
        flat = patches.reshape(bsz, num_h, num_w, ps * ps)

        # Quantize to histogram bins.
        idx = (flat * (bins / 256.0)).long().clamp(0, bins - 1)
        idx = idx.reshape(-1, ps * ps)
        one_hot = torch.zeros(idx.shape[0], ps * ps, bins, device=device, dtype=images.dtype)
        one_hot.scatter_(2, idx.unsqueeze(2), 1.0)
        hist = one_hot.sum(dim=1).reshape(bsz, num_h, num_w, bins)

        probs = hist / float(ps * ps)
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log2(probs + eps), dim=3)

        if pad_h > 0:
            entropy[:, -1, :] = pad_value
        if pad_w > 0:
            entropy[:, :, -1] = pad_value
        entropy_maps[ps] = entropy

    return entropy_maps


def select_patches_by_threshold(
    entropy_maps: Dict[int, torch.Tensor],
    thresholds: List[float],
) -> Dict[int, torch.Tensor]:
    """
    APT-style coarse-to-fine token selection.
    Returns 0/1 masks for each patch size.
    """
    patch_sizes = sorted(entropy_maps.keys())
    if len(patch_sizes) == 1:
        return {patch_sizes[0]: torch.ones_like(entropy_maps[patch_sizes[0]])}
    if len(thresholds) != len(patch_sizes) - 1:
        raise ValueError(
            f"Expected {len(patch_sizes) - 1} thresholds, got {len(thresholds)}"
        )

    masks: Dict[int, torch.Tensor] = {}
    base = patch_sizes[0]
    masks[base] = torch.ones_like(entropy_maps[base])

    for i in range(len(patch_sizes) - 1, 0, -1):
        ps = patch_sizes[i]
        thr = thresholds[i - 1]
        masks[ps] = (entropy_maps[ps] < thr).float()

    for i in range(len(patch_sizes) - 1, 0, -1):
        ps = patch_sizes[i]
        for j in range(i):
            smaller = patch_sizes[j]
            scale = ps // smaller
            up = masks[ps].repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
            h_small, w_small = entropy_maps[smaller].shape[1:]
            up = up[:, :h_small, :w_small]
            masks[smaller] = masks[smaller] * (1.0 - up)

    return masks


def build_base_keep_mask_from_entropy(
    raw_images: torch.Tensor,
    base_patch_size: int,
    num_scales: int,
    thresholds: Optional[List[float]],
    bins: int = 64,
) -> torch.Tensor:
    """
    Compute APT-style adaptive base-scale keep mask.
    Returns:
        keep mask [B, N_base], where 1 means keep fine token and 0 means merge-prior token.
    """
    if thresholds is None:
        thresholds = [4.5] * max(0, num_scales - 1)

    # Work in [0, 255] for entropy histogram.
    images = raw_images
    if images.dtype.is_floating_point:
        max_val = float(images.detach().amax().item())
        if max_val <= 1.5:
            images = images * 255.0
    images = images.clamp(0.0, 255.0)

    maps = compute_patch_entropy_batched(
        images=images,
        patch_size=base_patch_size,
        num_scales=num_scales,
        bins=bins,
    )
    masks = select_patches_by_threshold(maps, thresholds)
    keep = masks[base_patch_size].flatten(1)
    return keep
