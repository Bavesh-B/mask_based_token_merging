"""Convert the PanNuke parquet (RationAI mirror) to COCO instance format.

folds 1+2 -> train, fold 3 -> val (the standard PanNuke protocol, same as EoMT).
5 thing classes. Run with the maskmerge env python (needs pyarrow)."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils

CATS = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate(CATS)]


def convert(source, image_dir, folds, out_json):
    image_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    img_id, ann_id = 0, 0
    for fold in folds:
        t = pq.read_table(source / f"fold{fold}.parquet",
                          columns=["image", "instances", "categories"])
        imgs = t["image"].to_pylist()
        insts = t["instances"].to_pylist()
        catss = t["categories"].to_pylist()
        for row_idx, (imstruct, inst_list, cats) in enumerate(zip(imgs, insts, catss)):
            if len(inst_list) == 0:
                continue
            im = Image.open(io.BytesIO(imstruct["bytes"])).convert("RGB")
            W, H = im.size
            fn = f"f{fold}_{row_idx:05d}.png"
            im.save(image_dir / fn)
            img_id += 1
            images.append({"id": img_id, "file_name": fn, "height": H, "width": W})
            for inst, cat in zip(inst_list, cats):
                m = np.asarray(Image.open(io.BytesIO(inst["bytes"])), dtype=np.uint8)
                if m.sum() == 0:
                    continue
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                ann_id += 1
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": int(cat) + 1,
                    "segmentation": rle, "area": float(mask_utils.area(rle)),
                    "bbox": mask_utils.toBbox(rle).tolist(), "iscrowd": 0,
                })
    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"{out_json}: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for split, folds in (("train", [1, 2]), ("val", [3])):
        convert(args.source, args.output / "images", folds,
                args.output / "annotations" / f"instances_{split}.json")
