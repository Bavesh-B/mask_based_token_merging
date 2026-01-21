import math
from typing import Callable, Tuple
import torch


def do_nothing(x, mode=None):
    return x


def turbo_matching(
    metric: torch.Tensor,
    layer_idx: int,
    source: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
    num_prefix_tokens: int = 1,
) -> Tuple[Callable, Callable]:
    
    
    protected = num_prefix_tokens

    prefix = metric[..., :protected, :]
    metric = metric[..., protected:, :]

    t = metric.shape[1]
    r = t // 2

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)


        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # ------------------ start  addaptive section --------- 
        i = layer_idx
        n_B, n_H = node_max.shape
        node_mean= torch.add(node_max[:,1:].mean(dim=1).mean(),node_max[:,1:].std(dim=1).mean()/i)
        node_mean=node_mean.repeat(1,n_H)
        r = torch.ge(node_max, node_mean).sum(dim=1).min()

        # ------------------ end addaptive section --------- 
        
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        prefix_tokens = x[..., :protected, :]
        x = x[..., protected:, :]
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        merged = torch.cat([unm, dst], dim=1)
        return torch.cat([prefix_tokens, merged], dim=1)

    return merge





