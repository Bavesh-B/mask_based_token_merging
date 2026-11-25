"""Cityscapes instance segmentation as an EoMT instance dataset.

Cityscapes is a crowded urban benchmark (dense adjacent cars / pedestrians) --
the natural-image analog of the twin-dense nuclei regime, and the test of
whether mask-aware merging's advantage generalises beyond histology (ADE/COCO
are sparse and did NOT show it).

8 thing classes (hasInstances=True): person, rider, car, truck, bus, train,
motorcycle, bicycle. Instance IDs in gtFine `*_gtFine_instanceIds.png` encode
`class_id * 1000 + instance_index` for things (ids < 1000 are stuff, skipped).
Images from `leftImg8bit`. No crowd flags in Cityscapes.

Trained at 640x640 (crops) so the 40x40 = 1600-token grid and every merge
schedule measured on PanNuke/ADE transfer 1:1. Val returns native-resolution
2-tuples (eval_step resizes), like COCO/PanNuke.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

# Cityscapes id (hasInstances) -> contiguous label
THING_ID_TO_LABEL = {
    24: 0,   # person
    25: 1,   # rider
    26: 2,   # car
    27: 3,   # truck
    28: 4,   # bus
    31: 5,   # train
    32: 6,   # motorcycle
    33: 7,   # bicycle
}
CLASSES = ["person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]


class _CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, min_instances=1,
                 val_size=None):
        self.transforms = transforms
        # Downsize val images+masks to a fixed size (all methods identical) so
        # native-res (1024x2048) mask-mAP over 500 crowded images stays
        # tractable; the comparison is relative so a consistent smaller size
        # is fair. Training uses random 640 crops (transforms) as usual.
        self.val_size = val_size
        img_root = Path(root, "leftImg8bit", split)
        gt_root = Path(root, "gtFine", split)
        self.samples: list[tuple[Path, Path]] = []
        for city_dir in sorted(img_root.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_path.name.replace("_leftImg8bit.png", "")
                inst = gt_root / city_dir.name / f"{stem}_gtFine_instanceIds.png"
                if inst.exists():
                    self.samples.append((img_path, inst))
        self.min_instances = min_instances

    def __len__(self):
        return len(self.samples)

    def _parse(self, inst_path):
        inst = np.array(Image.open(inst_path))
        masks, labels = [], []
        for iid in np.unique(inst):
            if iid < 1000:  # stuff / non-instance
                continue
            cls_id = int(iid) // 1000
            if cls_id not in THING_ID_TO_LABEL:
                continue
            masks.append(torch.from_numpy(inst == iid))
            labels.append(THING_ID_TO_LABEL[cls_id])
        return masks, labels

    def __getitem__(self, index):
        img_path, inst_path = self.samples[index]
        try:
            img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        except OSError:
            # Rare corrupt/truncated PNG in the dataset -- skip to the next
            # sample instead of crashing the whole run.
            return self.__getitem__((index + 1) % len(self))
        masks, labels = self._parse(inst_path)
        if not masks:  # no thing instances: retry next (rare)
            return self.__getitem__((index + 1) % len(self))
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.zeros(len(labels), dtype=torch.bool),
        }
        if self.transforms is not None:
            return self.transforms(img, target)
        if self.val_size is not None:
            import torchvision.transforms.v2.functional as TF
            h, w = self.val_size
            img = TF.resize(img, [h, w])
            target["masks"] = tv_tensors.Mask(
                TF.resize(target["masks"], [h, w],
                          interpolation=TF.InterpolationMode.NEAREST)
            )
            keep = target["masks"].flatten(1).any(1)
            target = {k: v[keep] for k, v in target.items()}
        return img, target


class CityscapesInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 8,
        batch_size: int = 4,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 8,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        val_size: tuple[int, int] = None,
    ) -> None:
        super().__init__(
            path=path, batch_size=batch_size, num_workers=num_workers,
            num_classes=num_classes, img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.img_size = img_size
        self.val_size = tuple(val_size) if val_size is not None else None
        self.train_transforms = Transforms(
            img_size=img_size, color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None) -> "CityscapesInstance":
        self.train_dataset = _CityscapesDataset(
            self.path, "train", self.train_transforms)
        self.val_dataset = _CityscapesDataset(self.path, "val", None, val_size=self.val_size)
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, drop_last=True,
            collate_fn=self.train_collate, **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
