"""Albumentations-based augmentation pipeline for BraTS 2D slices.

All transforms are spatial (applied identically to image and mask) or
intensity-only (applied to image channels only).
"""

from __future__ import annotations

import albumentations as A
import numpy as np


def get_train_transforms(image_size: int = 240) -> A.Compose:
    """Standard augmentation pipeline for training."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            # Intensity augmentations on image channels only
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_limit=(0.01, 0.05), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
    )


def get_val_transforms(image_size: int = 240) -> None:
    """No augmentation at validation — return None to signal pass-through."""
    return None


class AlbumentationsWrapper:
    """Wraps an albumentations Compose to be compatible with BraTSDataset transform.

    BraTSDataset passes (image: np.ndarray (4,H,W), mask: np.ndarray (H,W))
    to transform. Albumentations expects HWC image and HW mask.
    """

    def __init__(self, transforms: A.Compose | None) -> None:
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.transforms is None:
            return image, mask

        # (4,H,W) → (H,W,4) for albumentations
        img_hwc = image.transpose(1, 2, 0)
        result = self.transforms(image=img_hwc, mask=mask)
        # Back to (4,H,W)
        return result["image"].transpose(2, 0, 1), result["mask"]
