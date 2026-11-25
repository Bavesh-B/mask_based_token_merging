"""GFLOPs/image for the PanNuke Mask2Former under dense / faithful-ALGM /
faithful-MaskMerge(2-pass). Reuses algm_full_probe's patches (import-time) for the
merge methods, then runs torch FlopCounterMode on a single real val image.
Usage: --method {dense,algm,maskmerge}."""
import sys

method = "dense"
if "--method" in sys.argv:
    i = sys.argv.index("--method"); method = sys.argv[i + 1]; del sys.argv[i:i + 2]

# strip our extra flag from argv before detectron2 parses; keep config/opts
if method in ("algm", "maskmerge"):
    sys.argv += ["--local-thresh", "0.945"]
    if method == "maskmerge":
        sys.argv += ["--gated"]
    import algm_full_probe  # noqa: F401  applies backbone (+ 2pass) patches, imports train_net
    import train_net
else:
    import register_pannuke  # noqa: F401
    import train_net

import torch
from torch.utils.flop_counter import FlopCounterMode
from detectron2.engine import default_argument_parser
from detectron2.data import build_detection_test_loader

args = default_argument_parser().parse_args()
cfg = train_net.setup(args)
model = train_net.Trainer.build_model(cfg)
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()

loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
total, n = 0.0, 0
with torch.no_grad():
    for batch in loader:
        with FlopCounterMode(display=False) as fc:
            model(batch)
        total += fc.get_total_flops() / 1e9
        n += 1
        if n >= 12:
            break
print(f"[flops_m2f] method={method}: {total / n:.1f} GFLOPs/image (mean over {n})")
