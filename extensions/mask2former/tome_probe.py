"""Backbone-merge probe: enable ToMe (bipartite COSINE token merging) in the
DINOv2 ViT backbone at eval, and measure PanNuke instance AP. This is the true
EoMT/ALGM analog -- the merge happens IN the image encoder, and (verified) the
unmerge copies the AVERAGED feature back to each position, so the quantization
reaches mask_features -> the readout -> can fuse adjacent twins.

No masks exist in the backbone, so this is COSINE only (the mask-aware fix would
need a 2-pass scheme). Usage: set the merge schedule via --tome-block/--tome-r,
enable ToMe via MODEL.DINOV2_CTS.ENABLE_TOME True + TOME_IN_EVAL True in opts.
New file; monkeypatches only the ToMeConfig schedule constant at runtime."""
import sys

block, r = 6, 288
if "--tome-block" in sys.argv:
    i = sys.argv.index("--tome-block"); block = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
if "--tome-r" in sys.argv:
    i = sys.argv.index("--tome-r"); r = int(sys.argv[i + 1]); del sys.argv[i:i + 2]

import mask2former.modeling.backbone.dinov2_cts as D
# DATASET="coco" in the config selects TOME_MERGE_SCHEDULE_COCO; patch it.
D.ToMeConfig.TOME_MERGE_SCHEDULE_COCO = [(block, r)]
print(f"[tome_probe] ToMe merge schedule = [({block}, {r})] (merge r={r} tokens at block {block})")

_ALGM = {"tau": None, "merged": 0.0, "total": 0.0, "calls": 0}
if "--tome-threshold" in sys.argv:
    i = sys.argv.index("--tome-threshold"); _ALGM["tau"] = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
    import torch
    import torch.nn.functional as F
    def _threshold_merge_tokens(self, tokens, r_cap):
        # ALGM-like GLOBAL merge: merge only pairs with cosine > tau (adaptive count),
        # instead of a forced top-r. This is the training-free, seg-validated criterion.
        B, N, C = tokens.shape
        if N < 2:
            return tokens, None
        a, b = tokens[:, 0::2, :], tokens[:, 1::2, :]
        n_a, n_b = a.shape[1], b.shape[1]
        if n_b == 0:
            return tokens, None
        an = F.normalize(a, dim=-1); bn = F.normalize(b, dim=-1)
        sim = an @ bn.transpose(-2, -1)          # [B, n_a, n_b]
        node_max = sim.max(dim=1)[0]             # [B, n_b] best-a similarity per b
        r = int((node_max > _ALGM["tau"]).sum(dim=1).min().item())   # adaptive (B=1 at eval)
        r = min(r, n_b, r_cap)
        _ALGM["merged"] += r; _ALGM["total"] += N; _ALGM["calls"] += 1
        if r <= 0:
            return tokens, None
        _, dst_idx = torch.topk(node_max, k=r, dim=1)
        src_idx = sim.max(dim=1)[1]
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1)
        src_idx = src_idx[batch_idx, dst_idx]
        merged = (a[batch_idx, src_idx] + b[batch_idx, dst_idx]) / 2
        a_keep = torch.ones(B, n_a, dtype=torch.bool, device=tokens.device); a_keep.scatter_(1, src_idx, False)
        b_keep = torch.ones(B, n_b, dtype=torch.bool, device=tokens.device); b_keep.scatter_(1, dst_idx, False)
        a_keep_idx = torch.argsort(a_keep.int(), dim=1, descending=True)[:, : n_a - r]
        b_keep_idx = torch.argsort(b_keep.int(), dim=1, descending=True)[:, : n_b - r]
        unmerged_a = torch.gather(a, 1, a_keep_idx.unsqueeze(-1).expand(-1, -1, C))
        unmerged_b = torch.gather(b, 1, b_keep_idx.unsqueeze(-1).expand(-1, -1, C))
        state = {"N": N, "r": r, "n_a": n_a, "n_b": n_b, "src_idx": src_idx, "dst_idx": dst_idx}
        return torch.cat([unmerged_a, unmerged_b, merged], dim=1), state
    D.ToMeReducer.merge_tokens = _threshold_merge_tokens
    import atexit
    atexit.register(lambda: print(f"[tome_probe] ALGM-threshold tau={_ALGM['tau']}: mean reduce="
                                  f"{_ALGM['merged']/max(_ALGM['total'],1):.3f} over {_ALGM['calls']} calls"))
    print(f"[tome_probe] ALGM-like THRESHOLD merge active (tau={_ALGM['tau']}, adaptive count)")

if "--tome-random" in sys.argv:
    sys.argv.remove("--tome-random")
    import torch
    def _random_merge_tokens(self, tokens, r):
        # RANDOM control: same ToMe merge mechanics, but random matching (not cosine).
        B, N, C = tokens.shape
        if r <= 0 or N < 2:
            return tokens, None
        a, b = tokens[:, 0::2, :], tokens[:, 1::2, :]
        n_a, n_b = a.shape[1], b.shape[1]
        if n_b == 0:
            return tokens, None
        r = min(r, n_b)
        if r <= 0:
            return tokens, None
        sim = torch.rand(B, n_a, n_b, device=tokens.device)  # <-- random instead of cosine
        node_max = sim.max(dim=1)[0]
        _, dst_idx = torch.topk(node_max, k=r, dim=1)
        src_idx = sim.max(dim=1)[1]
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1)
        src_idx = src_idx[batch_idx, dst_idx]
        merged = (a[batch_idx, src_idx] + b[batch_idx, dst_idx]) / 2
        a_keep = torch.ones(B, n_a, dtype=torch.bool, device=tokens.device); a_keep.scatter_(1, src_idx, False)
        b_keep = torch.ones(B, n_b, dtype=torch.bool, device=tokens.device); b_keep.scatter_(1, dst_idx, False)
        a_keep_idx = torch.argsort(a_keep.int(), dim=1, descending=True)[:, : n_a - r]
        b_keep_idx = torch.argsort(b_keep.int(), dim=1, descending=True)[:, : n_b - r]
        unmerged_a = torch.gather(a, 1, a_keep_idx.unsqueeze(-1).expand(-1, -1, C))
        unmerged_b = torch.gather(b, 1, b_keep_idx.unsqueeze(-1).expand(-1, -1, C))
        state = {"N": N, "r": r, "n_a": n_a, "n_b": n_b, "src_idx": src_idx, "dst_idx": dst_idx}
        return torch.cat([unmerged_a, unmerged_b, merged], dim=1), state
    D.ToMeReducer.merge_tokens = _random_merge_tokens
    print("[tome_probe] RANDOM merge control active (matching = random, not cosine)")

import register_pannuke  # noqa: F401  (register pannuke_coco_*)
import train_net
from detectron2.engine import launch, default_argument_parser

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("tome_probe args:", args)
    launch(train_net.main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
