"""Fix A -- train a lightweight coarse-instance head on backbone tokens, distilled
from the frozen Mask2Former's decoder masks. At inference this head gives per-token
instance identity IN THE SINGLE forward pass, so mask-gated merging needs no 2nd pass.

Head: 2 conv layers on the block-L ViT tokens -> Q-way per-token logits. Target: the
frozen model's decoder pred_masks argmax-over-queries, downsampled to the ViT grid.
Only the head trains (model frozen). Usage: --layer L --iters N --head-out PATH."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER, ITERS, HEAD_OUT = 9, 3000, "coarse_head.pth"
if "--layer" in sys.argv:
    i = sys.argv.index("--layer"); LAYER = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
if "--iters" in sys.argv:
    i = sys.argv.index("--iters"); ITERS = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
if "--head-out" in sys.argv:
    i = sys.argv.index("--head-out"); HEAD_OUT = sys.argv[i + 1]; del sys.argv[i:i + 2]

import register_pannuke  # noqa: F401
import train_net
import mask2former.modeling.backbone.dinov2_cts as D
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader
from detectron2.structures import ImageList


class CoarseHead(nn.Module):
    def __init__(self, cin=768, chid=256, Q=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, chid, 3, padding=1), nn.GroupNorm(32, chid), nn.ReLU(),
            nn.Conv2d(chid, chid, 3, padding=1), nn.GroupNorm(32, chid), nn.ReLU(),
            nn.Conv2d(chid, Q, 1))

    def forward(self, x):
        return self.net(x)


# dense backbone forward that stores the block-L spatial tokens on self._blockL
def _bb_forward_store(self, x):
    npx = self.num_prefix_tokens
    tokens = self.vit.patch_embed(x)
    rope = self._get_rope(x)
    if hasattr(self.vit, "_pos_embed"):
        tokens = self.vit._pos_embed(tokens)
    H = x.shape[2] // self.vit.patch_embed.patch_size[0]
    W = x.shape[3] // self.vit.patch_embed.patch_size[1]
    for i, block in enumerate(self.vit.blocks):
        tokens = self._block_forward(block, tokens, rope)
        if i == LAYER:
            self._blockL = tokens[:, npx:, :]
            self._gh, self._gw = H, W
    spatial = self.vit.norm(tokens[:, npx:, :])
    B = spatial.shape[0]
    return self.ms_proj(spatial.transpose(1, 2).reshape(B, -1, H, W))


D.D2DINOv2WithCTS.forward = _bb_forward_store

args = default_argument_parser().parse_args()
cfg = train_net.setup(args)
model = train_net.Trainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
for p in model.parameters():
    p.requires_grad = False
device = model.device
Q = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
cin = getattr(model.backbone, "embed_dim", None) or model.backbone.vit.embed_dim
head = CoarseHead(cin, 256, Q).to(device).train()
print(f"[coarse_head] cin={cin}", flush=True)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

loader = build_detection_train_loader(cfg)
print(f"[coarse_head] training head at block {LAYER}, Q={Q}, {ITERS} iters", flush=True)
it = 0
run = 0.0
for batch in loader:
    imgs = [(x["image"].to(device).float() - model.pixel_mean) / model.pixel_std for x in batch]
    images = ImageList.from_tensors(imgs, model.size_divisibility)
    with torch.no_grad():
        feats = model.backbone(images.tensor)
        out = model.sem_seg_head(feats)
        pm = out["pred_masks"]                                  # [B,Q,h,w]
    gh, gw = model.backbone._gh, model.backbone._gw
    tgt = F.interpolate(pm, size=(gh, gw), mode="bilinear", align_corners=False).argmax(1)  # [B,gh,gw]
    sp = model.backbone._blockL                                # [B,N,C]
    grid = sp.transpose(1, 2).reshape(sp.shape[0], -1, gh, gw)
    logits = head(grid.float())                                # [B,Q,gh,gw]
    loss = F.cross_entropy(logits, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    run += loss.item(); it += 1
    if it % 100 == 0:
        # agreement with target argmax (how well the head recovers the instance map)
        acc = (logits.argmax(1) == tgt).float().mean().item()
        print(f"[coarse_head] iter {it} loss {run/100:.3f} argmax-acc {acc:.3f}", flush=True)
        run = 0.0
    if it >= ITERS:
        break
torch.save({"state_dict": head.state_dict(), "layer": LAYER, "Q": Q}, HEAD_OUT)
print(f"[coarse_head] saved -> {HEAD_OUT}", flush=True)
