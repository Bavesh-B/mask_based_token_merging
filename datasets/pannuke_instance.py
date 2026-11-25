"""PanNuke nuclei instance segmentation as an EoMT instance dataset.

PanNuke (RationAI/PanNuke parquet mirror) is 7,901 H&E patches, 256x256, with
per-instance boolean masks and 5 nuclei categories (Neoplastic, Inflammatory,
Connective, Dead, Epithelial). Every category is a "thing" -- pure instance
segmentation, no stuff.

Why this dataset (CASCADE.md / PUBLICATION_CLAIM.md): nuclei fields are the
twin-dense regime as the NORM -- hundreds of adjacent, appearance-identical
instances per image -- where feature-cosine merging is predicted to fuse
instances (the 3.5-4x adjacent-vs-isolated loss signature measured on COCO).
Nuclei boundaries are also stain-crisp, so image-gradient walls have a real
shot here where ADE20K texture defeats them. This is the arena to test which
merge signal (cosine / gated / Sobel wall / MaskMerge / oracle) wins when
cosine structurally fails.

Images are upscaled to `img_size` (default 640x640) so the 40x40=1600 token
grid -- and therefore every cascade schedule and token budget already
measured -- transfers unchanged.

Standard protocol: train on folds 1+2, evaluate on fold 3.

The parquet rows are held as raw encoded bytes (~0.8 GB, shared copy-on-write
across forked workers) and decoded per __getitem__.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

CATEGORIES = [
    "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial",
]


class _PanNukeDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_paths, transforms, min_instances: int = 1):
        self.transforms = transforms
        self.rows: list[tuple[bytes, list[bytes], list[int]]] = []
        for p in parquet_paths:
            table = pq.read_table(
                p, columns=["image", "instances", "categories"]
            )
            images = table["image"].to_pylist()
            instances = table["instances"].to_pylist()
            categories = table["categories"].to_pylist()
            for img_struct, inst_list, cats in zip(images, instances, categories):
                if len(inst_list) < min_instances:
                    continue  # skip empty patches
                self.rows.append((
                    img_struct["bytes"],
                    [inst["bytes"] for inst in inst_list],
                    list(cats),
                ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        img_bytes, inst_bytes, cats = self.rows[index]
        img = tv_tensors.Image(
            Image.open(io.BytesIO(img_bytes)).convert("RGB")
        )
        masks = torch.stack([
            torch.from_numpy(np.array(Image.open(io.BytesIO(b)), dtype=bool))
            for b in inst_bytes
        ])
        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": torch.tensor(cats, dtype=torch.long),
            "is_crowd": torch.zeros(len(cats), dtype=torch.bool),
        }
        if self.transforms is not None:
            # train path: Transforms always returns (img, target, entropy|None);
            # train_collate consumes the 3-tuple.
            return self.transforms(img, target)
        # val path: native-resolution (img, target); MaskClassificationInstance
        # .eval_step resizes/pads each image internally, exactly as for COCO.
        return img, target


class PanNukeInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 8,
        batch_size: int = 8,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 5,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        train_folds: tuple[int, ...] = (1, 2),
        val_folds: tuple[int, ...] = (3,),
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.train_folds = tuple(train_folds)
        self.val_folds = tuple(val_folds)
        self.img_size = img_size

        # Train transforms: scale jitter + crop + colour jitter (as COCO).
        # Val transforms: deterministic resize to img_size via ScaleJitter at
        # scale 1 range is not identity, so build a plain resize path instead.
        self.train_transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        # val: no transforms -- eval_step resizes/pads natively, as for COCO.
        self.val_transforms = None

    def _paths(self, folds) -> list[Path]:
        return [Path(self.path, f"fold{f}.parquet") for f in folds]

    def setup(self, stage: Union[str, None] = None) -> "PanNukeInstance":
        self.train_dataset = _PanNukeDataset(
            self._paths(self.train_folds), self.train_transforms
        )
        self.val_dataset = _PanNukeDataset(
            self._paths(self.val_folds), self.val_transforms
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
