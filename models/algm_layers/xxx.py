import math
from typing import Callable, Optional, Tuple
import torch

def do_nothing(x, mode=None):
    return x
def merge_wavg(merge: Callable, x: torch.Tensor, size: Optional[torch.Tensor] = None):
    """Weighted-average merge as in ToMe."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])  # [B,T,1]
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size
def merge_source(merge: Callable, x: torch.Tensor, source: Optional[torch.Tensor] = None):
    """Adjacency/source tracking as in ToMe."""
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    source = merge(source, mode="amax")
    return source
# -------------------------
# Global ToMe (bipartite) WITH mask+confidence gating
# -------------------------
def bipartite_soft_matching_masked(
    metric: torch.Tensor,       # [B,T,D]
    P: torch.Tensor,            # [B,Q,Npatch] mask probs for PATCH TOKENS ONLY
    tau: float,
    r: int = math.inf,
    class_token: bool = True,
    num_query_tokens: int = 0,  # protected query tokens at the end
) -> Tuple[Callable, Callable]:
    """
    Global bipartite ToMe with:
      - protect CLS (if class_token)
      - protect query tokens (assumed last num_query_tokens)
      - only allow merges among PATCH TOKENS (and only when mask/conf pass)
    Input token layout assumed: [CLS] + [PATCH TOKENS] + [QUERY TOKENS (optional)]
    """
    B, T, D = metric.shape
    protected = 0
    if class_token:
        protected += 1
    protected += int(num_query_tokens)
    if T - protected <= 1:
        return do_nothing, do_nothing
    # Determine patch token span
    # Layout: [0:cls] [1:1+Npatch] [1+Npatch: end queries]
    # So Npatch = T - protected
    Npatch = T - protected
    assert P.shape[0] == B and P.shape[2] == Npatch, f"P must be [B,Q,{Npatch}] for current token count."
    # We can reduce by max 50% of merge-eligible tokens (like ToMe)
    r = min(r, (Npatch // 2))
    if r <= 0:
        return do_nothing, do_nothing
    with torch.no_grad():
        # Split out the portion we actually merge over: patch tokens only
        m_patch = metric[:, (1 if class_token else 0):(1 if class_token else 0) + Npatch, :]  # [B,Npatch,D]
        # Mask stats (top1 id + confidence) per PATCH token
        conf, gid = P.max(dim=1)  # [B,Npatch], [B,Npatch]
        # Normalize metric for cosine similarity
        m_patch = m_patch / (m_patch.norm(dim=-1, keepdim=True) + 1e-12)
        # bipartite split over patch tokens in original order: even/odd
        a, b = m_patch[..., ::2, :], m_patch[..., 1::2, :]      # [B,Na,D], [B,Nb,D]
        scores = a @ b.transpose(-1, -2)                         # [B,Na,Nb]
        # Apply constraints (mask + conf)
        a_conf = conf[..., ::2]
        b_conf = conf[..., 1::2]
        a_gid  = gid[..., ::2]
        b_gid  = gid[..., 1::2]
        ok = (a_conf >= tau).unsqueeze(-1) & (b_conf >= tau).unsqueeze(-2) & (a_gid.unsqueeze(-1) == b_gid.unsqueeze(-2))
        scores = scores.masked_fill(~ok, -math.inf)
        # Standard ToMe selection
        node_max, node_idx = scores.max(dim=-1)                         # [B,Na]
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # [B,Na,1]
        unm_idx = edge_idx[..., r:, :]                                   # [B,Na-r,1]
        src_idx = edge_idx[..., :r, :]                                   # [B,r,1]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)      # [B,r,1]
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # Token layout: [CLS] + [PATCH] + [QUERY]
        cls_part = x[:, :1, :] if class_token else x[:, :0, :]
        q_part = x[:, -num_query_tokens:, :] if num_query_tokens > 0 else x[:, :0, :]
        patch = x[:, (1 if class_token else 0):(1 if class_token else 0) + Npatch, :]
        src, dst = patch[..., ::2, :], patch[..., 1::2, :]
        n, na, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, na - r, c))
        src_m = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_m, reduce=mode)
        patch_out = torch.cat([unm, dst], dim=1)
        return torch.cat([cls_part, patch_out, q_part], dim=1)
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        cls_part = x[:, :1, :] if class_token else x[:, :0, :]
        q_part = x[:, -num_query_tokens:, :] if num_query_tokens > 0 else x[:, :0, :]
        # Split patch back
        patch_out = x[:, (1 if class_token else 0): x.shape[1] - num_query_tokens, :]
        unm_len = unm_idx.shape[1]
        unm, dst = patch_out[:, :unm_len, :], patch_out[:, unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out_patch = torch.zeros(n, Npatch, c, device=x.device, dtype=x.dtype)
        out_patch[..., 1::2, :] = dst
        out_patch.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out_patch.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return torch.cat([cls_part, out_patch, q_part], dim=1)
    return merge, unmerge