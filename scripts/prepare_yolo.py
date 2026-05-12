"""Convert prepared BraTS slices into a YOLO one-class tumor dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.data.yolo import write_yolo_sample
from src.utils.config import load_config


def main(config_path: str, output: str, margin: int = 8) -> None:
    cfg = load_config(config_path)
    split = load_split(cfg["data"]["split_file"])
    root = Path(output)
    counts = {}
    for split_name, patient_ids in split.items():
        ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=patient_ids)
        saved = 0
        for idx in range(len(ds)):
            image, mask = ds[idx]
            image_path = root / "images" / split_name / f"{idx:06d}.png"
            label_path = root / "labels" / split_name / f"{idx:06d}.txt"
            if write_yolo_sample(image.numpy(), mask.numpy(), image_path, label_path, margin=margin):
                saved += 1
        counts[split_name] = saved

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: tumor\n",
        encoding="utf-8",
    )
    print(f"YOLO dataset saved to {root}. Tumor-positive counts: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/yolo_tumor")
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()
    main(args.config, args.output, args.margin)
