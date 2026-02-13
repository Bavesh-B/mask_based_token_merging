import math
from typing import Callable, Tuple
import torch


def do_nothing(x, mode=None):
    return x


def _l1_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.sum(dim=-1, keepdim=True) + eps)


def _pairwise_js_similarity(
    a_probs: torch.Tensor,
    b_probs: torch.Tensor,
    eps: float = 1e-12,
    chunk_size: int = 128,
) -> torch.Tensor:
    # a_probs: [B, Na, Q], b_probs: [B, Nb, Q]
    a_probs = _l1_normalize(a_probs, eps=eps).clamp_min(eps)
    b_probs = _l1_normalize(b_probs, eps=eps).clamp_min(eps)
    a_log = a_probs.log()
    b_log = b_probs.log()

    bsz, na, _ = a_probs.shape
    nb = b_probs.shape[1]
    out = a_probs.new_empty((bsz, na, nb))
    log2 = math.log(2.0)

    for start in range(0, na, chunk_size):
        end = min(start + chunk_size, na)
        a = a_probs[:, start:end, :].unsqueeze(2)  # [B, C, 1, Q]
        b = b_probs.unsqueeze(1)  # [B, 1, Nb, Q]
        m = (a + b) * 0.5
        m_log = m.clamp_min(eps).log()

        a_log_chunk = a_log[:, start:end, :].unsqueeze(2)
        b_log_full = b_log.unsqueeze(1)
        kl_pm = (a * (a_log_chunk - m_log)).sum(dim=-1)
        kl_qm = (b * (b_log_full - m_log)).sum(dim=-1)
        js = 0.5 * (kl_pm + kl_qm)
        out[:, start:end, :] = 1.0 - (js / log2)

    return out


def _mask_distribution_scores(
    metric: torch.Tensor,
    mask_probs: torch.Tensor,
    score_metric: str = "js",
) -> torch.Tensor:
    a_probs = mask_probs[..., ::2].transpose(1, 2)  # [B, N/2, Q]
    b_probs = mask_probs[..., 1::2].transpose(1, 2)  # [B, N/2, Q]
    eps = 1e-12

    if score_metric == "js":
        return _pairwise_js_similarity(a_probs, b_probs, eps=eps)
    if score_metric == "hellinger":
        a = _l1_normalize(a_probs, eps=eps).clamp_min(eps).sqrt()
        b = _l1_normalize(b_probs, eps=eps).clamp_min(eps).sqrt()
        return a @ b.transpose(-1, -2)
    if score_metric == "dot":
        a = _l1_normalize(a_probs, eps=eps)
        b = _l1_normalize(b_probs, eps=eps)
        return a @ b.transpose(-1, -2)
    if score_metric == "cosine":
        a = metric[..., ::2, :]
        b = metric[..., 1::2, :]
        a = a / (a.norm(dim=-1, keepdim=True) + eps)
        b = b / (b.norm(dim=-1, keepdim=True) + eps)
        return a @ b.transpose(-1, -2)
    raise ValueError(
        "score_metric must be one of: 'cosine', 'js', 'hellinger', 'dot'"
    )


def turbo_matching(
    metric: torch.Tensor,
    layer_idx:int,
    source: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
    num_prefix_tokens: int = 1,
) -> Tuple[Callable, Callable]:
    
    
    protected = 0
    # if class_token:
    #     # print('CLASS TOKEN!!', protected)
    #     protected += 1
    if distill_token:
        # print('DISTILL TOKEN!!', protected)
        protected += 1
    protected += num_prefix_tokens

    t = metric.shape[1]
    r = (t - protected) // 2
    # print('R!!', r, t, protected, (t - protected))

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():

        B,m_t,um_t = source.shape
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
       
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        if num_prefix_tokens > 0:
            scores[..., :num_prefix_tokens, :] = -math.inf
            scores[..., :, :num_prefix_tokens] = -math.inf


        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # ------------------ start  addaptive section --------- 
        i = layer_idx
        n_B, n_H = node_max.shape
        node_mean= torch.add(node_max[:,num_prefix_tokens:].mean(dim=1).mean(),node_max[:,num_prefix_tokens:].std(dim=1).mean()/i)
        node_mean=node_mean.repeat(1,n_H)
        r = torch.ge(node_max, node_mean).sum(dim=1).min()

        # ------------------ end addaptive section --------- 
        
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token or num_prefix_tokens > 0:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

        r=int(r.item())


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            # print('CONCATTTTT!!', torch.cat([unm, dst], dim=1).shape, unm.shape, dst.shape, unm_idx.shape, src_idx.shape, dst_idx.shape, edge_idx.shape, scores.shape)
            # print('NEXT LINE!!', r, metric.shape, source.shape)
            return torch.cat([unm, dst], dim=1)

    return merge


def global_bipartite_matching_masked(
    metric: torch.Tensor,
    mask_probs: torch.Tensor,
    layer_idx: int,
    tau: float = 0.0,
    score_metric: str = "js",
) -> Tuple[Callable, dict]:
    """
    Global bipartite matching using mask similarity only.
    Merges are gated by confidence.
    """
    bsz, _, num_patch = mask_probs.shape
    if num_patch <= 1:
        return do_nothing, None

    with torch.no_grad():
        conf, _ = mask_probs.max(dim=1)  # [B, Npatch]

        # Token-token scores from mask distributions (or cosine baseline).
        scores = _mask_distribution_scores(
            metric=metric,
            mask_probs=mask_probs,
            score_metric=score_metric,
        )

        a_conf = conf[..., ::2]
        b_conf = conf[..., 1::2]
        ok = (
            (a_conf >= tau).unsqueeze(-1)
            & (b_conf >= tau).unsqueeze(-2)
        )
        scores = scores.masked_fill(~ok, -math.inf)

        node_max, node_idx = scores.max(dim=-1)
        valid = torch.isfinite(node_max)
        if not valid.any():
            return do_nothing, None

        valid_counts = valid.sum(dim=1).clamp(min=1)
        node_max_masked = node_max.masked_fill(~valid, 0.0)
        mean = node_max_masked.sum(dim=1) / valid_counts
        var = ((node_max_masked - mean.unsqueeze(-1)) ** 2 * valid).sum(dim=1) / valid_counts
        std = var.sqrt()
        denom = max(int(layer_idx), 1)
        node_mean = mean + std / denom

        r = ((node_max >= node_mean.unsqueeze(-1)) & valid).sum(dim=1).min()
        r = int(r.item())
        if r <= 0:
            return do_nothing, None

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    info = {
        "unm_idx": unm_idx,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "num_patch": num_patch,
    }

    return merge, info

