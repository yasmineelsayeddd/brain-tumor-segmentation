"""Augmentation pipeline for BraTS 2D slices."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import albumentations as A
except ImportError:  # Keep imports/test discovery usable before Kaggle deps install.
    A = None


def get_train_transforms(image_size: int = 240) -> Any:
    """Standard augmentation pipeline for training.

    Albumentations is optional at import time so local smoke tests can run before
    installing the full Kaggle stack. Training with augmentations still requires
    the package.
    """
    if A is None:
        raise ImportError("albumentations is required for training augmentations. Install requirements.txt first.")
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
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_limit=(0.01, 0.05), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
    )


def get_val_transforms(image_size: int = 240) -> None:
    """No augmentation at validation."""
    return None


class AlbumentationsWrapper:
    """Wrap an albumentations Compose for (C,H,W) image and (H,W) mask arrays."""

    def __init__(self, transforms: Any | None) -> None:
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.transforms is None:
            return image, mask
        # cv2-backed transforms (GaussianBlur, ElasticTransform, ShiftScaleRotate, …)
        # require float32. Cast here; BraTSDataset converts back to tensor.float() anyway.
        img_hwc = image.transpose(1, 2, 0).astype(np.float32)
        result = self.transforms(image=img_hwc, mask=mask)
        return result["image"].transpose(2, 0, 1), result["mask"]
