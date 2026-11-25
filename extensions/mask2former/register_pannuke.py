"""Register the converted PanNuke COCO instance datasets for detectron2.
Import this module before building the model/trainer. New file; touches nothing."""
import os
from pathlib import Path

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

_ROOT = Path(os.environ.get("PANNUKE_COCO_ROOT", "datasets/pannuke_coco")).resolve()
_THING = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
for _split in ["train", "val"]:
    _name = f"pannuke_coco_{_split}"
    if _name not in DatasetCatalog.list():
        register_coco_instances(
            _name, {},
            str(_ROOT / "annotations" / f"instances_{_split}.json"),
            str(_ROOT / "images"),
        )
        MetadataCatalog.get(_name).thing_classes = _THING
