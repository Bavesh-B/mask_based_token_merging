"""MaskClassificationInstance that honours the network's
``allowed_checkpoint_missing_prefixes`` (exactly as TaskTransportPanoptic
does), so transport-family networks -- whose routing heads do not exist in a
dense checkpoint -- can be evaluated zero-shot on COCO instance segmentation.

New module rather than an edit: the inherited loader raises on ANY missing
key outside its hardcoded entropy prefixes."""

from __future__ import annotations

from training.mask_classification_instance import MaskClassificationInstance


class IdeasMaskClassificationInstance(MaskClassificationInstance):
    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        allowed = [
            "network.entropy_patch_attn.",
            "network.entropy_zero_mlp.",
        ] + [
            f"network.{prefix}"
            for prefix in getattr(
                self.network, "allowed_checkpoint_missing_prefixes", ()
            )
        ]
        missing = list(incompatible_keys.missing_keys)
        if not load_ckpt_class_head:
            missing = [
                k for k in missing
                if "class_head" not in k and "class_predictor" not in k
            ]
        missing = [
            k for k in missing
            if not any(k.startswith(p) for p in allowed)
        ]
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        if incompatible_keys.unexpected_keys:
            raise ValueError(
                f"Unexpected keys: {incompatible_keys.unexpected_keys}"
            )
