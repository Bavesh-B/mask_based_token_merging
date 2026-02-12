# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import lightning


class LightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        num_classes: int,
        check_empty_targets: bool,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        self.path = path
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        self.img_size = img_size
        self.num_classes = num_classes

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def train_collate(batch):
        imgs, targets = [], []
        entropy_maps_batch = []

        for sample in batch:
            if len(sample) == 3:
                img, target, entropy_maps = sample
                entropy_maps_batch.append(entropy_maps)
            else:
                img, target = sample
            imgs.append(img)
            targets.append(target)

        imgs = torch.stack(imgs)
        if not entropy_maps_batch:
            return imgs, targets

        patch_sizes = sorted(entropy_maps_batch[0].keys())
        batched_maps = {
            ps: torch.stack([em[ps] for em in entropy_maps_batch], dim=0)
            for ps in patch_sizes
        }
        return imgs, targets, batched_maps

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))
