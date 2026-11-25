"""Fix A -- SINGLE-PASS mask-gated merging via the distilled coarse head.

At block L the head predicts a per-token coarse instance map from the (dense) ViT
tokens, IN THE SAME forward pass. That map gates the global adaptive bipartite merge
at block L (block cross-instance src/dst pairs). No 2nd pass. So MaskMerge now costs
one forward (like ALGM), not two.

Modes:
  --gated        use the head's instance map to gate (MaskMerge). Omit -> ungated (ALGM).
  --thr-scale S  scale the adaptive threshold (lower S -> more merges -> more reduction).
  --head PATH    coarse-head checkpoint (needed only with --gated).
  --flops        measure GFLOPs/image instead of running the AP eval.
Reuses the EoMT ALGM helpers (merge_source/merge_wavg) and algm_full_probe's unmerge."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.algm_layers.local_merge import merge_source, merge_wavg

LAYER, GATED, THR_SCALE, FLOPS = 6, False, 1.0, False
HEAD_PATH = "coarse_head.pth"
for flag, cast in (("--layer", int), ("--thr-scale", float), ("--head", str)):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        if flag == "--layer": LAYER = v
        elif flag == "--thr-scale": THR_SCALE = v
        else: HEAD_PATH = v
if "--gated" in sys.argv:
    sys.argv.remove("--gated"); GATED = True
if "--flops" in sys.argv:
    sys.argv.remove("--flops"); FLOPS = True
TWOPASS = False
if "--twopass" in sys.argv:       # gate with the decoder's REAL masks (2-pass) = ceiling
    sys.argv.remove("--twopass"); TWOPASS = True; GATED = True
BENCH = False
if "--bench" in sys.argv:
    sys.argv.remove("--bench"); BENCH = True
DENSE = False
if "--dense" in sys.argv:          # no merge at all (throughput reference)
    sys.argv.remove("--dense"); DENSE = True

G = {"before": 0.0, "after": 0.0, "calls": 0}
DEC = {"inst": None}


class CoarseHead(nn.Module):
    def __init__(self, cin=768, chid=256, Q=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, chid, 3, padding=1), nn.GroupNorm(32, chid), nn.ReLU(),
            nn.Conv2d(chid, chid, 3, padding=1), nn.GroupNorm(32, chid), nn.ReLU(),
            nn.Conv2d(chid, Q, 1))

    def forward(self, x):
        return self.net(x)


def _do_nothing(x, mode=None):
    return x


def _global_matching(metric, layer_idx, num_prefix, inst=None, thr_scale=1.0):
    protected = int(num_prefix)
    vm = metric[:, protected:, :]
    n = vm.shape[1]
    if n <= 1:
        return _do_nothing
    with torch.no_grad():
        mn = vm / (vm.norm(dim=-1, keepdim=True) + 1e-12)
        src_m, dst_m = mn[..., ::2, :], mn[..., 1::2, :]
        if src_m.shape[1] == 0 or dst_m.shape[1] == 0:
            return _do_nothing
        scores = src_m @ dst_m.transpose(-1, -2)
        # ALGM's own plan: each src -> its best dst, threshold from the (ungated)
        # similarity distribution. The gate NEVER changes this plan or the threshold;
        # it only CANCELS the merges whose src/dst straddle an instance boundary.
        node_max, node_idx = scores.max(dim=-1)
        denom = max(int(layer_idx), 1)
        ts = node_max[:, 1:] if node_max.shape[1] > 1 else node_max
        thr = (ts.mean(dim=1).mean() + ts.std(dim=1, unbiased=False).mean() / denom) * thr_scale
        keep = torch.ge(node_max, thr)                       # ALGM would merge these src
        if inst is not None and inst.numel() == n:
            a_i, b_i = inst[::2], inst[1::2]                 # src / dst instance ids
            dst_inst = b_i[node_idx]                         # instance of each src's best dst
            keep = keep & (a_i.view(1, -1) == dst_inst)      # cancel cross-instance merges
        r = int(keep.sum(dim=1).min().item())
        if r <= 0:
            return _do_nothing
        eff = torch.where(keep, node_max, node_max.new_full((), -1e9))
        edge_idx = eff.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]; src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="mean"):
        pfx = x[:, :protected, :]; vis = x[:, protected:, :]
        src, dst = vis[..., ::2, :], vis[..., 1::2, :]
        B, sl, C = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(B, sl - r, C))
        src = src.gather(dim=-2, index=src_idx.expand(B, r, C))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)
        return torch.cat((pfx, unm, dst), dim=1)
    return merge


def _unmerge_to_grid(x, source, num_prefix):
    sv = source[:, num_prefix:, num_prefix:]
    vis = x[:, num_prefix:, :]
    idxs = sv.argmax(dim=1)
    full = torch.gather(vis, 1, idxs.unsqueeze(-1).expand(-1, idxs.shape[1], vis.shape[2]))
    return torch.cat((x[:, :num_prefix, :], full), dim=1)


import mask2former.modeling.backbone.dinov2_cts as D

_HEAD = {"m": None}


def _headgated_backbone_forward(self, x):
    npx = self.num_prefix_tokens
    tokens = self.vit.patch_embed(x)
    rope = self._get_rope(x)
    if hasattr(self.vit, "_pos_embed"):
        tokens = self.vit._pos_embed(tokens)
    H = x.shape[2] // self.vit.patch_embed.patch_size[0]
    W = x.shape[3] // self.vit.patch_embed.patch_size[1]
    source = None; size = None
    n0 = tokens.shape[1] - npx
    for i, block in enumerate(self.vit.blocks):
        tokens = self._block_forward(block, tokens, rope)
        if i == LAYER:
            if TWOPASS and DEC["inst"] is None:
                continue                                      # pass-1 dense: no merge
            inst = None
            if TWOPASS:
                inst = DEC["inst"]                            # decoder's real masks (ceiling)
            elif GATED:
                sp = tokens[:, npx:, :]                       # dense grid, no prior merge
                B, N, C = sp.shape
                grid = sp.transpose(1, 2).reshape(B, C, H, W)
                with torch.no_grad():
                    inst = _HEAD["m"](grid.float()).argmax(1).reshape(-1)
            merge = _global_matching(tokens, layer_idx=i, num_prefix=npx,
                                     inst=inst, thr_scale=THR_SCALE)
            source = merge_source(merge, tokens, source)
            tokens, size = merge_wavg(merge, tokens, size)
    n1 = tokens.shape[1] - npx
    if not (TWOPASS and DEC["inst"] is None):   # don't count the dense pass-1 of a 2-pass
        G["before"] += n0; G["after"] += n1; G["calls"] += 1
    if source is not None:
        tokens = _unmerge_to_grid(tokens, source, npx)
    spatial = self.vit.norm(tokens[:, npx:, :])
    B = spatial.shape[0]
    return self.ms_proj(spatial.transpose(1, 2).reshape(B, -1, H, W))


if not DENSE:
    D.D2DINOv2WithCTS.forward = _headgated_backbone_forward

if TWOPASS:
    from mask2former.maskformer_model import MaskFormer
    from detectron2.structures import ImageList
    from detectron2.modeling.postprocessing import sem_seg_postprocess
    from detectron2.utils.memory import retry_if_cuda_oom

    def _2pass_forward(self, batched_inputs):
        images = [(x["image"].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        Ht, Wt = images.tensor.shape[-2:]
        ph = self.backbone.vit.patch_embed.patch_size[0]
        gh, gw = Ht // ph, Wt // ph
        DEC["inst"] = None                                   # pass 1: dense
        out1 = self.sem_seg_head(self.backbone(images.tensor))
        pm = out1["pred_masks"][0]
        DEC["inst"] = F.interpolate(pm.argmax(0).float()[None, None], size=(gh, gw),
                                    mode="nearest")[0, 0].long().reshape(-1)
        outputs = self.sem_seg_head(self.backbone(images.tensor))   # pass 2: gated merge
        DEC["inst"] = None
        mc = outputs["pred_logits"]
        mp = F.interpolate(outputs["pred_masks"], size=(Ht, Wt), mode="bilinear", align_corners=False)
        res = []
        for c, p, inp, isz in zip(mc, mp, batched_inputs, images.image_sizes):
            h = inp.get("height", isz[0]); w = inp.get("width", isz[1]); res.append({})
            if self.sem_seg_postprocess_before_inference:
                p = retry_if_cuda_oom(sem_seg_postprocess)(p, isz, h, w); c = c.to(p)
            if self.instance_on:
                res[-1]["instances"] = retry_if_cuda_oom(self.instance_inference)(c, p)
        return res
    MaskFormer.forward = _2pass_forward

import atexit
atexit.register(lambda: print(f"[coarse_eval] gated={GATED} L={LAYER} thr_scale={THR_SCALE} "
                              f"mean_reduce={1 - G['after']/max(G['before'],1):.3f} over {G['calls']} calls"))

import register_pannuke  # noqa: F401
import train_net
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer


def _load_head(model):
    if not GATED or TWOPASS:
        return
    ck = torch.load(HEAD_PATH, map_location="cpu")
    Q = ck.get("Q", 200)
    cin = getattr(model.backbone, "embed_dim", None) or model.backbone.vit.embed_dim
    h = CoarseHead(cin, 256, Q)
    h.load_state_dict(ck["state_dict"])
    _HEAD["m"] = h.to(model.device).eval()
    print(f"[coarse_eval] loaded head L={ck.get('layer')} Q={Q} from {HEAD_PATH}", flush=True)


def _bench(model, cfg, n_warm=20, n_time=300):
    import time
    from detectron2.data import build_detection_test_loader
    loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= n_warm + n_time:
            break
    with torch.no_grad():
        for b in batches[:n_warm]:
            model(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for b in batches[n_warm:n_warm + n_time]:
            model(b)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    m = len(batches) - n_warm
    print(f"[bench] dense={DENSE} gated={GATED} twopass={TWOPASS} L={LAYER} thr={THR_SCALE}: "
          f"{m/dt:.1f} img/s ({1000*dt/m:.2f} ms/img over {m})")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if BENCH:
        cfg = train_net.setup(args)
        model = train_net.Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval(); _load_head(model)
        _bench(model, cfg)
    elif FLOPS:
        from torch.utils.flop_counter import FlopCounterMode
        from detectron2.data import build_detection_test_loader
        cfg = train_net.setup(args)
        model = train_net.Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval(); _load_head(model)
        loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                with FlopCounterMode(display=False) as fc:
                    model(batch)
                total += fc.get_total_flops() / 1e9; n += 1
                if n >= 12:
                    break
        print(f"[coarse_eval] GATED={GATED} L={LAYER}: {total / n:.1f} GFLOPs/image (mean over {n})")
    else:
        # patch build_model path so the head loads before eval
        _orig = train_net.Trainer.build_model.__func__

        def _bm(cls, cfg):
            m = _orig(cls, cfg)
            _load_head(m)
            return m
        train_net.Trainer.build_model = classmethod(_bm)
        train_net.main(args)
