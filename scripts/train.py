"""Train a segmentation model.

Usage (local):
    python scripts/train.py --config configs/default.yaml

Usage (Kaggle — set paths in notebook, then call main()):
    from scripts.train import main; main()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.models.unet import UNet
from src.training.augmentation import AlbumentationsWrapper, get_train_transforms, get_val_transforms
from src.training.losses import DiceCELoss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    split = load_split(cfg["data"]["split_file"])
    train_ds = BraTSDataset(
        cfg["data"]["data_root"],
        patient_ids=split["train"],
        transform=AlbumentationsWrapper(get_train_transforms(cfg["data"]["image_size"])),
    )
    val_ds = BraTSDataset(
        cfg["data"]["data_root"],
        patient_ids=split["val"],
        transform=None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
    )
    print(f"Model parameters: {model.parameter_count():,}")

    criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("min_lr", 1e-6),
    )

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=cfg["model"]["out_channels"],
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        experiment_name=cfg["experiment"]["name"],
    )

    print(f"\nBest val Dice: {trainer.best_val_dice:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
