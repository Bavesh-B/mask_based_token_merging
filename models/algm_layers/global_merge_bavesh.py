import math
from typing import Callable, Tuple
import torch


def do_nothing(x, mode=None):
    return x


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





