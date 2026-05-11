"""Segmentation evaluation metrics.

All functions operate on numpy arrays or torch tensors.
Multi-class metrics return per-class values (excluding background by default).
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dice_per_class(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
    ignore_background: bool = True,
) -> np.ndarray:
    """Compute Dice coefficient per class.

    Args:
        pred: (H, W) predicted class indices.
        gt:   (H, W) ground-truth class indices.

    Returns:
        Array of length (num_classes) or (num_classes - 1) if ignore_background.
    """
    pred = _to_numpy(pred)
    gt   = _to_numpy(gt)

    start = 1 if ignore_background else 0
    scores = []
    for c in range(start, num_classes):
        p = (pred == c)
        g = (gt   == c)
        intersection = (p & g).sum()
        scores.append((2 * intersection + smooth) / (p.sum() + g.sum() + smooth))
    return np.array(scores, dtype=np.float32)


def iou_per_class(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
    ignore_background: bool = True,
) -> np.ndarray:
    """Intersection-over-Union per class."""
    pred = _to_numpy(pred)
    gt   = _to_numpy(gt)

    start = 1 if ignore_background else 0
    scores = []
    for c in range(start, num_classes):
        p = (pred == c)
        g = (gt   == c)
        intersection = (p & g).sum()
        union = (p | g).sum()
        scores.append((intersection + smooth) / (union + smooth))
    return np.array(scores, dtype=np.float32)


def mean_dice(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    ignore_background: bool = True,
) -> float:
    return float(dice_per_class(pred, gt, num_classes, ignore_background=ignore_background).mean())


def mean_iou(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    ignore_background: bool = True,
) -> float:
    return float(iou_per_class(pred, gt, num_classes, ignore_background=ignore_background).mean())


def pixel_accuracy(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
) -> float:
    pred = _to_numpy(pred)
    gt   = _to_numpy(gt)
    return float((pred == gt).mean())
