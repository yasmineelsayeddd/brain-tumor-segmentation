"""Train BraTS segmentation models on Modal — persistent volumes, no session limits.

One-time local setup:
    pip install modal
    modal token new          # opens a browser to authenticate

Step 1 — pull the prepared dataset from Kaggle into a persistent volume (run once, ~15 min):
    modal run --detach modal_train.py::download_data

Step 2 — upload the 3 resume checkpoints into the checkpoint volume.
         (the remote name must be exactly "<experiment_name>_best.pth" — set it explicitly,
          so it doesn't matter if your local file is named "... (1).pth")
    modal volume put brats-checkpoints "C:/Users/yasmi/Downloads/attention_unet_best.pth"          /attention_unet_best.pth
    modal volume put brats-checkpoints "C:/Users/yasmi/Downloads/uncertainty_unet_best.pth"        /uncertainty_unet_best.pth
    modal volume put brats-checkpoints "C:/Users/yasmi/Downloads/unet_best_fusion_loss_aug_best.pth" /unet_best_fusion_loss_aug_best.pth

Step 3 — train. Each model auto-resumes if a checkpoint for it already exists on the volume.
    modal run --detach modal_train.py --all                 # all 5 models in parallel
    modal run --detach modal_train.py --config unetpp.yaml  # or just one

Step 4 — when jobs finish, pull the checkpoints + training history back to your machine:
    modal volume get brats-checkpoints / ./modal_checkpoints

Track running jobs at https://modal.com/apps
"""

from __future__ import annotations

from pathlib import Path

import modal

# Kaggle API token (same one the notebooks use; the dataset is public).
KAGGLE_TOKEN = "KGAT_bb60250db72735c2a11893aa4a1e0db7"
DATASET_SLUG = "yasmineelqorashy/brats2020-2d-prepared"

# default.yaml -> unet_baseline, unetpp.yaml -> unetpp : both train from scratch.
# The other three auto-resume from the checkpoints uploaded in step 2.
CONFIGS = [
    "default.yaml",
    "unetpp.yaml",
    "unet_best_fusion_loss_aug.yaml",
    "attention_unet.yaml",
    "uncertainty_unet.yaml",
]

LOCAL_REPO = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "opencv-python-headless",
        "albumentations",
        "Pillow",
        "pyyaml",
        "tqdm",
        "tensorboard",
        "kaggle",
    )
    .add_local_dir(LOCAL_REPO / "src", "/root/project/src")
    .add_local_dir(LOCAL_REPO / "scripts", "/root/project/scripts")
    .add_local_dir(LOCAL_REPO / "configs", "/root/project/configs")
)

app = modal.App("brats-segmentation")

data_vol = modal.Volume.from_name("brats-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("brats-checkpoints", create_if_missing=True)


@app.function(image=image, volumes={"/data": data_vol}, timeout=3600)
def download_data() -> None:
    """One-time: download the prepared BraTS dataset from Kaggle into the data volume."""
    import os
    import subprocess

    existing = list(Path("/data").rglob("metadata.json"))
    if existing:
        print(f"Dataset already on volume: {existing[0].parent}")
        return

    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    print(f"Downloading {DATASET_SLUG} (~12 GB) ...")
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", "/data", "--unzip"]
    )
    data_vol.commit()
    meta = [str(m) for m in Path("/data").rglob("metadata.json")]
    print(f"Done. metadata.json found at: {meta}")


@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=8.0,
    memory=16384,
    timeout=86400,  # 24 h ceiling — a single model finishes long before this
)
def train_model(config_name: str) -> None:
    """Train one model. Auto-resumes from the checkpoint volume if a checkpoint exists."""
    import os
    import subprocess
    import sys

    import yaml

    project = Path("/root/project")
    os.chdir(project)

    # Locate the dataset on the data volume.
    data_vol.reload()
    try:
        meta = next(Path("/data").rglob("metadata.json"))
        split = next(Path("/data").rglob("default.json"))
    except StopIteration as exc:
        raise RuntimeError(
            "Dataset not found on volume — run `modal run modal_train.py::download_data` first."
        ) from exc
    data_root = meta.parent

    # Load the repo config and patch paths to this run's volume locations.
    with open(project / "configs" / config_name) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = str(data_root)
    cfg["data"]["split_file"] = str(split)
    cfg["checkpoint_dir"] = "/ckpt"
    cfg["experiment"]["output_dir"] = "/ckpt/outputs"

    runtime_cfg = Path("/tmp/runtime_config.yaml")
    with open(runtime_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    exp_name = cfg["experiment"]["name"]

    # Auto-resume: prefer the full-state _last.pth, fall back to an uploaded _best.pth.
    ckpt_vol.reload()
    resume_arg: list[str] = []
    last = Path("/ckpt") / f"{exp_name}_last.pth"
    best = Path("/ckpt") / f"{exp_name}_best.pth"
    if last.exists():
        resume_arg = ["--resume", str(last)]
        print(f"[{exp_name}] resuming from {last.name}")
    elif best.exists():
        resume_arg = ["--resume", str(best)]
        print(f"[{exp_name}] resuming from {best.name} (first resume)")
    else:
        print(f"[{exp_name}] training from scratch")

    cmd = [sys.executable, "-m", "scripts.train", "--config", str(runtime_cfg), *resume_arg]
    subprocess.check_call(cmd, cwd=str(project))
    ckpt_vol.commit()
    print(f"[{exp_name}] finished — checkpoint saved to the brats-checkpoints volume.")


@app.local_entrypoint()
def main(config: str = "", all: bool = False) -> None:
    if all:
        handles = [train_model.spawn(c) for c in CONFIGS]
        print(f"Spawned {len(handles)} training jobs — running in parallel on Modal.")
    elif config:
        train_model.spawn(config)
        print(f"Spawned training for {config}.")
    else:
        print("Specify --config <name>.yaml  or  --all")
        return
    print("Jobs run server-side. Track them at https://modal.com/apps")
