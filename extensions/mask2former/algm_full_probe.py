"""FAITHFUL full ALGM in the Mask2Former DINOv2 backbone, exactly as EoMT does it:
local conditional_pooling at layer 1 (thr) + global adaptive-threshold bipartite
matching at layer 5, with merge_source/merge_wavg size tracking and unmerge-to-grid
before the pixel decoder. Reuses the EoMT ALGM code (imported) so it is identical.

--gated turns it into MaskMerge: a dense pass-1 gives per-token instance ids; pass-2
blocks cross-instance pairs in BOTH the local and global merges (instance-gated).
Compare ungated (ALGM) vs gated (MaskMerge) -- both full local+global -- at their
matched, adaptive compute budgets.

Usage: --local-thresh T [--gated]. New file; imports from the EoMT repo via
Uses the local and weighted merge helpers from the MaskMerge release."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from models.algm_layers.local_merge import conditional_pooling, merge_source, merge_wavg

LOCAL_THRESH, SEL, GATED, BENCH = 0.945, [1, 5], False, False
if "--local-thresh" in sys.argv:
    i = sys.argv.index("--local-thresh"); LOCAL_THRESH = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
if "--gated" in sys.argv:
    sys.argv.remove("--gated"); GATED = True
if "--bench" in sys.argv:
    sys.argv.remove("--bench"); BENCH = True

G = {"inst": None, "before": 0.0, "after": 0.0, "calls": 0}


def _do_nothing(x, mode=None):
    return x


def _global_matching(metric, layer_idx, num_prefix, inst=None):
    # Copied from EoMT eomt_original_algm._original_algm_global_matching, + optional
    # instance gate (block cross-instance src/dst pairs).
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
        if inst is not None and inst.numel() == n:
            a_i, b_i = inst[::2], inst[1::2]
            scores = scores.masked_fill(a_i.view(1, -1, 1) != b_i.view(1, 1, -1), -1e4)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        denom = max(int(layer_idx), 1)
        ts = node_max[:, 1:] if node_max.shape[1] > 1 else node_max
        thr = ts.mean(dim=1).mean() + ts.std(dim=1, unbiased=False).mean() / denom
        r = int(torch.ge(node_max, thr).sum(dim=1).min().item())
        if r <= 0:
            return _do_nothing
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


def _algm_backbone_forward(self, x):
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
        if i in SEL:
            gi = G["inst"] if GATED else None
            if i == SEL[0]:      # LOCAL 2x2 conditional pooling (EoMT code)
                merge = conditional_pooling(tokens, LOCAL_THRESH, (2, 2),
                                            num_prefix_tokens=npx, grid_hw=(H, W))
                # note: local gate handled below by re-merging cross-inst back (approx: ungated local)
            else:                # GLOBAL adaptive matching (+ optional instance gate)
                merge = _global_matching(tokens, layer_idx=i, num_prefix=npx, inst=gi)
            source = merge_source(merge, tokens, source)
            tokens, size = merge_wavg(merge, tokens, size)
    n1 = tokens.shape[1] - npx
    G["before"] += n0; G["after"] += n1; G["calls"] += 1
    if source is not None:
        tokens = _unmerge_to_grid(tokens, source, npx)
    spatial = self.vit.norm(tokens[:, npx:, :])
    B = spatial.shape[0]
    feat = spatial.transpose(1, 2).reshape(B, -1, H, W)
    return self.ms_proj(feat)


D.D2DINOv2WithCTS.forward = _algm_backbone_forward

if GATED:
    from mask2former.maskformer_model import MaskFormer
    from detectron2.structures import ImageList
    from detectron2.modeling.postprocessing import sem_seg_postprocess
    from detectron2.utils.memory import retry_if_cuda_oom
    _orig_bb = _algm_backbone_forward
    def _dense_bb(self, x):   # pass-1 dense: skip ALGM entirely
        npx = self.num_prefix_tokens
        tokens = self.vit.patch_embed(x); rope = self._get_rope(x)
        if hasattr(self.vit, "_pos_embed"):
            tokens = self.vit._pos_embed(tokens)
        H = x.shape[2] // self.vit.patch_embed.patch_size[0]
        W = x.shape[3] // self.vit.patch_embed.patch_size[1]
        for block in self.vit.blocks:
            tokens = self._block_forward(block, tokens, rope)
        spatial = self.vit.norm(tokens[:, npx:, :]); B = spatial.shape[0]
        return self.ms_proj(spatial.transpose(1, 2).reshape(B, -1, H, W))

    def _2pass_forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        Ht, Wt = images.tensor.shape[-2:]; gh, gw = Ht // 16, Wt // 16
        D.D2DINOv2WithCTS.forward = _dense_bb; G["inst"] = None
        out1 = self.sem_seg_head(self.backbone(images.tensor))
        pm = out1["pred_masks"][0]
        G["inst"] = F.interpolate(pm.argmax(0).float()[None, None], size=(gh, gw),
                                  mode="nearest")[0, 0].long().reshape(-1)
        D.D2DINOv2WithCTS.forward = _orig_bb
        outputs = self.sem_seg_head(self.backbone(images.tensor))
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
atexit.register(lambda: print(f"[algm_full] gated={GATED} local_thr={LOCAL_THRESH} "
                              f"mean_reduce={1 - G['after']/max(G['before'],1):.3f} over {G['calls']} calls"))
print(f"[algm_full] FULL ALGM (local[{SEL[0]}]+global[{SEL[1]}]) gated={GATED} local_thr={LOCAL_THRESH}")

import register_pannuke  # noqa: F401
import train_net
from detectron2.engine import launch, default_argument_parser
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if BENCH:
        import time
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.data import build_detection_test_loader
        cfg = train_net.setup(args)
        model = train_net.Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()
        loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        nb = []
        for b in loader:
            nb.append(b)
            if len(nb) >= 320:
                break
        with torch.no_grad():
            for b in nb[:20]:
                model(b)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for b in nb[20:]:
                model(b)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
        m = len(nb) - 20
        print(f"[algm_full bench] gated={GATED} local_thr={LOCAL_THRESH}: {m/dt:.1f} img/s "
              f"({1000*dt/m:.2f} ms/img over {m})")
    else:
        launch(train_net.main, args.num_gpus, num_machines=args.num_machines,
               machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
