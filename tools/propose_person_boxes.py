"""Fill missing *proposed* boxes with MediaPipe; never approves a frame or box."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from prototype.pose import PoseExtractor, bbox_from_landmarks, primary_pose
from prototype.schema import box, path_under, read_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    parser.add_argument("--pose-task", default="artifacts/pose_landmarker.task")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    path = path_under(ROOT, args.manifest)
    rows = read_manifest(path)
    proposed = failures = 0
    with PoseExtractor(path_under(ROOT, args.pose_task)) as extractor:
        for row in rows:
            if row["review_state"] != "pending":
                continue
            if box(row) is not None and not row["review_note"].startswith("MediaPipe person-box proposal"):
                continue
            image = cv2.imread(str(path_under(ROOT, row["image_path"])))
            if image is None:
                failures += 1
                continue
            selected, status = primary_pose(extractor.detect(image), image.shape[1], image.shape[0])
            if status != "ok":
                failures += 1
                continue
            values = bbox_from_landmarks(selected, image.shape[1], image.shape[0])
            if values is None:
                failures += 1
                continue
            for name, value in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), values):
                row[name] = str(value)
            row["review_note"] = "MediaPipe person-box proposal; reviewer must confirm or redraw"
            proposed += 1
            if proposed % 50 == 0:
                write_manifest(path, rows)
                print(f"Proposed {proposed} boxes", flush=True)
            if args.limit and proposed >= args.limit:
                break
    write_manifest(path, rows)
    print(f"Proposed {proposed} boxes; {failures} frames need a manually drawn box")


if __name__ == "__main__":
    main()
