"""Cache the official Ultralytics YOLO12n detection checkpoint before training."""
from __future__ import annotations

import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "artifacts/ultralytics_config"
CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG))

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from prototype.pose import sha256_file
from prototype.schema import save_json

def main():
    target = ROOT / "artifacts/yolo/yolo12n_pretrained.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        original = Path(attempt_download_asset("yolo12n.pt"))
        if not original.is_file():
            raise FileNotFoundError("Official yolo12n.pt asset could not be downloaded")
        shutil.copyfile(original, target)
    model = YOLO(str(target))
    if model.task != "detect":
        raise ValueError("Downloaded YOLO12n checkpoint is not a detection model")
    save_json(target.with_suffix(".source.json"), {"asset":"yolo12n.pt", "publisher":"Ultralytics",
                                                   "ultralytics":ultralytics.__version__, "sha256":sha256_file(target),
                                                   "bytes":target.stat().st_size})
    print(f"{target}: {target.stat().st_size} bytes SHA256={sha256_file(target)}")


if __name__ == "__main__":
    main()
