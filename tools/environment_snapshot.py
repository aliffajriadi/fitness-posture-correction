"""Record versions and pretrained hashes used by the local prototype."""
from __future__ import annotations

import platform
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"artifacts/ultralytics_config"
CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG))

import cv2
import mediapipe
import numpy
import sklearn
import torch
import ultralytics

from prototype.pose import sha256_file
from prototype.schema import save_json

def main():
    pose = ROOT/"artifacts/pose_landmarker.task"
    yolo = ROOT/"artifacts/yolo/yolo12n_pretrained.pt"
    save_json(ROOT/"manifests/prototype_environment.json", {
        "python":platform.python_version(), "windows":platform.platform(),
        "numpy":numpy.__version__, "opencv":cv2.__version__, "mediapipe":mediapipe.__version__,
        "scikit_learn":sklearn.__version__, "torch":torch.__version__,
        "torch_cuda_available":torch.cuda.is_available(), "ultralytics":ultralytics.__version__,
        "pose_task_sha256":sha256_file(pose) if pose.exists() else None,
        "yolo12n_pretrained_sha256":sha256_file(yolo) if yolo.exists() else None,
    })
    print("Wrote manifests/prototype_environment.json")


if __name__ == "__main__":
    main()
