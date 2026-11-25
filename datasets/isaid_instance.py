"""iSAID aerial instance segmentation as an EoMT instance dataset.

iSAID is the twin-dense AERIAL analog of PanNuke: overhead imagery packed with
adjacent, near-identical, same-class instances -- parking lots of Small_Vehicles
(84k instances), harbors of ships, storage-tank farms, airfields of planes.
Median 56 instances/image, up to 7436. This is the regime where appearance-only
(cosine) merging fuses twins and mask-aware merging should win, a THIRD domain
(histology + urban + aerial) for the generalization claim.

Images are the original full-size DOTA v1 tiles (up to ~13000x6000 px) with COCO
polygon instance annotations. Full-image rasterization would OOM (thousands of
instances), so -- matching the STANDARD iSAID protocol -- we operate on fixed
patches: random crops for training, a deterministic tile grid for val (each tile
is an independent eval image). Only instances intersecting a crop are rasterized,
on the crop canvas, so cost scales with crop content, not whole-image density.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

cv2.setNumThreads(0)  # avoid thread oversubscription inside DataLoader workers
from torch.utils.data import DataLoader
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as TF

from datasets.lightning_data_module import LightningDataModule

Image.MAX_IMAGE_PIXELS = None  # DOTA tiles exceed PIL's default bomb limit

CLASSES = ["ship", "storage_tank", "baseball_diamond", "tennis_court",
           "basketball_court", "ground_track_field", "bridge", "large_vehicle",
           "small_vehicle", "helicopter", "swimming_pool", "roundabout",
           "soccer_ball_field", "plane", "harbor"]
# CRITICAL: the iSAID train and val JSONs use DIFFERENT category_id orderings
# for the same class names (e.g. Large_Vehicle is id 2 in train but id 8 in
# val). So the id->label map MUST be derived per-file from the class NAME, not
# hardcoded by id. NAME_TO_LABEL is the single canonical (name -> contiguous)
# mapping; each split's cat2label is built from its own categories via names.
NAME_TO_LABEL = {n: i for i, n in enumerate(CLASSES)}


def _rasterize_anns(anns, x0, y0, out_w, out_h, cat2label, min_pixels, scale=1.0):
    """Rasterize the polygons of `anns` that intersect the window
    [x0, x0+out_w/scale) x [y0, y0+out_h/scale) onto an out_h x out_w canvas
    (polygon coords shifted by (x0,y0) and scaled). Returns (masks, labels)."""
    win_w, win_h = out_w / scale, out_h / scale
    masks, labels = [], []
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        if bx + bw < x0 or bx > x0 + win_w or by + bh < y0 or by > y0 + win_h:
            continue
        seg = a["segmentation"]
        if not isinstance(seg, list):  # skip RLE (iSAID is polygons)
            continue
        m = Image.new("L", (out_w, out_h), 0)
        dr = ImageDraw.Draw(m)
        drawn = False
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = [((poly[i] - x0) * scale, (poly[i + 1] - y0) * scale)
                   for i in range(0, len(poly), 2)]
            dr.polygon(pts, fill=1)
            drawn = True
        if not drawn:
            continue
        arr = np.asarray(m, dtype=bool)
        if arr.sum() < min_pixels:
            continue
        masks.append(torch.from_numpy(arr))
        labels.append(cat2label[a["category_id"]])
    return masks, labels


class _ISAIDDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, crop_size, training,
                 scale_range=(0.7, 1.5), min_pixels=16, val_overlap=0.0,
                 max_val_tiles_per_img=64):
        self.img_dir = Path(root, "images", split)
        self.coco = COCO(str(Path(root, "annotations", f"iSAID_{split}.json")))
        self.crop = int(crop_size)
        self.training = training
        self.scale_range = scale_range
        self.min_pixels = min_pixels
        # Build id->label from THIS file's categories, by name (train/val use
        # different id orderings for the same names).
        self.cat2label = {
            c["id"]: NAME_TO_LABEL[c["name"].lower()]
            for c in self.coco.dataset["categories"]
        }

        self.img_ids = [i for i in self.coco.getImgIds()
                        if len(self.coco.getAnnIds(imgIds=i)) > 0]

        if not training:
            # deterministic tile grid; keep only tiles containing >=1 instance
            self.tiles = []
            stride = max(1, int(self.crop * (1 - val_overlap)))
            for iid in self.img_ids:
                info = self.coco.imgs[iid]
                W, H = info["width"], info["height"]
                anns = self.coco.imgToAnns[iid]
                xs = self._starts(W, self.crop, stride)
                ys = self._starts(H, self.crop, stride)
                added = 0
                for y0 in ys:
                    for x0 in xs:
                        if any(not (a["bbox"][0] + a["bbox"][2] < x0
                                    or a["bbox"][0] > x0 + self.crop
                                    or a["bbox"][1] + a["bbox"][3] < y0
                                    or a["bbox"][1] > y0 + self.crop)
                               for a in anns):
                            self.tiles.append((iid, x0, y0))
                            added += 1
                            if added >= max_val_tiles_per_img:
                                break
                    if added >= max_val_tiles_per_img:
                        break

    @staticmethod
    def _starts(dim, tile, stride):
        if dim <= tile:
            return [0]
        pos = list(range(0, dim - tile + 1, stride))
        if pos[-1] != dim - tile:
            pos.append(dim - tile)
        return pos

    def __len__(self):
        return len(self.img_ids) if self.training else len(self.tiles)

    def _load_img_np(self, iid):
        # Decode with OpenCV (robust on the occasional giant DOTA PNG where PIL's
        # libpng intermittently segfaults inside DataLoader workers). Returns a
        # HxWx3 uint8 RGB array.
        info = self.coco.imgs[iid]
        arr = cv2.imread(str(self.img_dir / info["file_name"]), cv2.IMREAD_COLOR)
        if arr is None:  # unreadable -> signal caller to skip
            return None
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def _train_item(self, index):
        for _ in range(20):
            iid = self.img_ids[index]
            arr = self._load_img_np(iid)
            if arr is None:
                index = random.randint(0, len(self.img_ids) - 1)
                continue
            H, W = arr.shape[:2]
            scale = random.uniform(*self.scale_range)
            win = max(16, int(round(self.crop / scale)))
            win = min(win, W, H)
            x0 = random.randint(0, max(0, W - win))
            y0 = random.randint(0, max(0, H - win))
            crop = arr[y0:y0 + win, x0:x0 + win]
            crop = cv2.resize(crop, (self.crop, self.crop),
                              interpolation=cv2.INTER_LINEAR)
            eff = self.crop / win
            masks, labels = _rasterize_anns(
                self.coco.imgToAnns[iid], x0, y0, self.crop, self.crop,
                self.cat2label, self.min_pixels, scale=eff)
            if masks:
                img_t = tv_tensors.Image(
                    torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1))
                target = {
                    "masks": tv_tensors.Mask(torch.stack(masks)),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "is_crowd": torch.zeros(len(labels), dtype=torch.bool),
                }
                if random.random() < 0.5:  # hflip
                    img_t = TF.hflip(img_t)
                    target["masks"] = tv_tensors.Mask(TF.hflip(target["masks"]))
                return img_t, target
            index = random.randint(0, len(self.img_ids) - 1)
        return self._train_item(random.randint(0, len(self.img_ids) - 1))

    def _val_item(self, index):
        iid, x0, y0 = self.tiles[index]
        arr = self._load_img_np(iid)
        crop = np.zeros((self.crop, self.crop, 3), dtype=np.uint8)
        if arr is not None:
            sub = arr[y0:y0 + self.crop, x0:x0 + self.crop]
            crop[:sub.shape[0], :sub.shape[1]] = sub
        masks, labels = _rasterize_anns(
            self.coco.imgToAnns[iid], x0, y0, self.crop, self.crop,
            self.cat2label, self.min_pixels, scale=1.0)
        img_t = tv_tensors.Image(
            torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1))
        if not masks:
            target = {"masks": tv_tensors.Mask(torch.zeros((0, self.crop, self.crop), dtype=torch.bool)),
                      "labels": torch.zeros((0,), dtype=torch.long),
                      "is_crowd": torch.zeros((0,), dtype=torch.bool)}
        else:
            target = {"masks": tv_tensors.Mask(torch.stack(masks)),
                      "labels": torch.tensor(labels, dtype=torch.long),
                      "is_crowd": torch.zeros(len(labels), dtype=torch.bool)}
        return img_t, target

    def __getitem__(self, index):
        return self._train_item(index) if self.training else self._val_item(index)


class ISAIDInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 8,
        batch_size: int = 8,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 15,
        check_empty_targets: bool = True,
        scale_range: tuple[float, float] = (0.7, 1.5),
        val_overlap: float = 0.0,
    ) -> None:
        super().__init__(
            path=path, batch_size=batch_size, num_workers=num_workers,
            num_classes=num_classes, img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.img_size = img_size
        self.crop_size = img_size[0]
        self.scale_range = scale_range
        self.val_overlap = val_overlap

    def setup(self, stage: Union[str, None] = None) -> "ISAIDInstance":
        self.train_dataset = _ISAIDDataset(
            self.path, "train", self.crop_size, training=True,
            scale_range=self.scale_range)
        self.val_dataset = _ISAIDDataset(
            self.path, "val", self.crop_size, training=False,
            val_overlap=self.val_overlap)
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, drop_last=True,
            collate_fn=self.train_collate, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=self.eval_collate,
            **self.dataloader_kwargs)
