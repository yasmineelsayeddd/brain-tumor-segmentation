"""
One-command Kaggle code sync.

Usage:
    python push_kaggle.py              # build zip + upload new dataset version
    python push_kaggle.py --run 05B   # also trigger notebook run after upload
"""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "yasmineelsayeddd/brats-seg-code"
ZIP_NAME     = "brats-seg-code.zip"

EXCLUDE_DIRS  = {".git", ".claude", "data", "outputs", "checkpoints", "__pycache__", ".pytest_cache"}
EXCLUDE_EXTS  = {".pyc", ".zip", ".pdf"}
ALLOW_ROOTS   = {"src", "scripts", "configs", "notebooks", "tests", "requirements.txt", "requirements-dev.txt", ".gitignore"}

NOTEBOOK_SLUGS = {
    "05A": "yasmineelsayeddd/05a-kaggle-prepare-data",
    "05B": "yasmineelsayeddd/05b-kaggle-train-one-model",
    "05C": "yasmineelsayeddd/05c-kaggle-evaluate-and-classical",
    "05D": "yasmineelsayeddd/05d-kaggle-yolo-detector",
}


def build_zip(root: Path, dest: Path) -> None:
    dest.unlink(missing_ok=True)
    count = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            rel   = path.relative_to(root)
            parts = rel.parts
            if parts[0] not in ALLOW_ROOTS:
                continue
            if any(p in EXCLUDE_DIRS for p in parts):
                continue
            if path.is_file() and path.suffix not in EXCLUDE_EXTS:
                zf.write(path, "/".join(parts))
                count += 1
    print(f"  Built {dest.name}  ({dest.stat().st_size // 1024} KB, {count} files)")


def upload(zip_path: Path, message: str) -> None:
    tmp = zip_path.parent / "_kaggle_upload"
    tmp.mkdir(exist_ok=True)
    meta = tmp / "dataset-metadata.json"
    meta.write_text(json.dumps({
        "title": "brats-seg-code",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }))
    import shutil
    shutil.copy2(zip_path, tmp / zip_path.name)

    print("  Uploading to Kaggle …")
    # Try update first; if dataset doesn't exist yet, create it
    result = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(tmp), "-m", message, "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "404" in result.stderr:
        print("  Dataset not found — creating it …")
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(tmp), "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    print("  Upload done.")
    shutil.rmtree(tmp)


def run_notebook(name: str) -> None:
    slug = NOTEBOOK_SLUGS.get(name)
    if not slug:
        print(f"Unknown notebook '{name}'. Choices: {list(NOTEBOOK_SLUGS)}")
        return
    print(f"  Triggering {name} ({slug}) …")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-k", slug],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    else:
        print("  Notebook queued. Check: https://www.kaggle.com/code/" + slug)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",     metavar="NB",  help="Notebook to trigger after upload (05A/05B/05C/05D)")
    parser.add_argument("--message", default="update code", help="Version message")
    parser.add_argument("--zip-only", action="store_true", help="Only build zip, don't upload")
    args = parser.parse_args()

    root     = Path(__file__).parent
    zip_path = root / ZIP_NAME

    print("[1/3] Building zip …")
    build_zip(root, zip_path)

    if args.zip_only:
        print(f"Zip saved to {zip_path}")
        return

    print("[2/3] Uploading dataset version …")
    upload(zip_path, args.message)

    if args.run:
        print(f"[3/3] Running notebook {args.run} …")
        run_notebook(args.run)
    else:
        print("[3/3] Done. To also trigger a notebook run: python push_kaggle.py --run 05B")


if __name__ == "__main__":
    main()
