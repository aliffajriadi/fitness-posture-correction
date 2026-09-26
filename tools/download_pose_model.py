"""Download the official MediaPipe Pose Landmarker Lite task bundle."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from prototype.pose import sha256_file
from prototype.schema import save_json

ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/pose_landmarker.task")
    args = parser.parse_args()
    target = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT):
        raise SystemExit("Output must stay in workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(".task.partial")
        try:
            urllib.request.urlretrieve(URL, temp)
            if temp.stat().st_size < 100000:
                raise ValueError("Downloaded model unexpectedly small")
            temp.replace(target)
        finally:
            if temp.exists():
                temp.unlink()
    save_json(target.with_suffix(".source.json"), {"url":URL, "sha256":sha256_file(target), "bytes":target.stat().st_size,
                                                   "variant":"lite/float16/latest"})
    print(f"{target}: {target.stat().st_size} bytes SHA256={sha256_file(target)}")


if __name__ == "__main__":
    main()
