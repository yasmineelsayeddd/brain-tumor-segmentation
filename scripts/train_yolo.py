"""Train the one-class YOLO detector for the cascade."""

from __future__ import annotations

import argparse


def main(data_yaml: str, model: str = "yolov8n.pt", epochs: int = 50, imgsz: int = 240, project: str = "outputs/yolo") -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for YOLO training. Install requirements.txt first.") from exc

    yolo = YOLO(model)
    yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, project=project, name="tumor_detector")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", default="data/yolo_tumor/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=240)
    parser.add_argument("--project", default="outputs/yolo")
    args = parser.parse_args()
    main(args.data_yaml, args.model, args.epochs, args.imgsz, args.project)
