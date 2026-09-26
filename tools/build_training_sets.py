"""Validate reviewed data and export one common set of form frame IDs.

This prepares data only. It never infers labels from folder names.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from prototype.pose import PoseExtractor, SPEC_VERSION, bbox_from_landmarks, primary_pose, sha256_file, vectorize
from prototype.schema import EXERCISES, FORMS, FORM_LABELS, assert_no_leakage, box, encode_form_label, form_support, path_under, read_manifest, save_json, valid_rows

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb-intersection else 0


def export_yolo(rows, base, names):
    import yaml
    for row in rows:
        split = row["split"]
        image_path = path_under(ROOT, row["image_path"])
        target_image = base / split / "images" / f"{row['sample_id']}.jpg"
        target_label = base / split / "labels" / f"{row['sample_id']}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, target_image)
        x1, y1, x2, y2 = box(row)
        w, h = int(row["width"]), int(row["height"])
        label = row["exercise"] if len(names) == 5 else encode_form_label(row["exercise"], row["form"])
        class_id = names.index(label)
        target_label.write_text(f"{class_id} {(x1+x2)/(2*w):.7f} {(y1+y2)/(2*h):.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}\n", encoding="utf-8")
    cfg = {"path": str(base.resolve()), "train":"train/images", "val":"val/images", "test":"test/images", "names":dict(enumerate(names))}
    (base / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    parser.add_argument("--pose-task", required=True)
    args = parser.parse_args()
    path = path_under(ROOT, args.manifest)
    rows = read_manifest(path)
    valid_rows(rows, ROOT)
    assert_no_leakage(rows)
    accepted = [r for r in rows if r["review_state"] == "accepted"]
    if not accepted:
        raise SystemExit("No human-reviewed accepted frames; training export is blocked.")
    for source_path, expected in {(r["source_path"], r["source_sha256"]) for r in accepted}:
        if expected and sha256_file(path_under(ROOT, source_path)) != expected:
            raise ValueError(f"Source changed since audit: {source_path}")
    task_path = path_under(ROOT, args.pose_task)
    features, geometry, pose_status = {}, {}, {}
    with PoseExtractor(task_path) as extractor:
        for n, row in enumerate(accepted, 1):
            image = cv2.imread(str(path_under(ROOT, row["image_path"])))
            poses = extractor.detect(image)
            chosen, status = primary_pose(poses, image.shape[1], image.shape[0])
            if status == "ok" and iou(bbox_from_landmarks(chosen, image.shape[1], image.shape[0]), box(row)) < .20:
                status = "bbox_mismatch"
            pose_status[row["sample_id"]] = status
            if status == "ok":
                features[row["sample_id"]], geometry[row["sample_id"]] = vectorize(chosen)
            if n % 50 == 0:
                print(f"Pose extracted {n}/{len(accepted)}", flush=True)
    out = ROOT / "data/derived/training"
    out.mkdir(parents=True, exist_ok=True)
    feature_ids = sorted(features)
    if not feature_ids:
        raise SystemExit("All accepted frames failed single-person pose extraction; no features to export")
    np.savez_compressed(out / "features.npz", ids=np.asarray(feature_ids), features=np.stack([features[i] for i in feature_ids]))
    save_json(out / "geometry.json", geometry)
    pose_ok = set(features)
    # Exactly these frame IDs train both form branches; extraction failures stay in evaluation.
    form_rows = [r for r in accepted if r["form"] in FORMS and r["sample_id"] in pose_ok]
    recognition_rows = [r for r in accepted if r["sample_id"] in pose_ok]
    support = form_support(form_rows, split="train")
    enabled = [ex for ex in EXERCISES if support[ex]["enabled"]]
    form_rows = [r for r in form_rows if r["exercise"] in enabled]
    # YOLO exercise detector may use pose-failure frames; form data is paired by ID.
    export_yolo(accepted, out / "yolo_exercise", list(EXERCISES))
    if form_rows:
        export_yolo(form_rows, out / "yolo_form", list(FORM_LABELS))
    index = {"manifest": str(path.relative_to(ROOT)), "manifest_sha256": sha256_file(path),
             "pose_model": str(task_path.relative_to(ROOT)), "pose_model_sha256": sha256_file(task_path),
             "feature_spec": SPEC_VERSION, "pose_status": pose_status,
             "recognition_train_ids": [r["sample_id"] for r in recognition_rows if r["split"] == "train"],
             "form_training_ids": [r["sample_id"] for r in form_rows if r["split"] == "train"],
             "form_ids_by_split": {s:[r["sample_id"] for r in form_rows if r["split"] == s] for s in ("train", "val", "test")},
             "form_support": support, "enabled_form": enabled, "exercise_labels": list(EXERCISES),
             "form_labels": list(FORM_LABELS),
             "counts": {"accepted":len(accepted), "pose_ok":len(pose_ok), "pose_status":dict(Counter(pose_status.values()))}}
    save_json(out / "training_index.json", index)
    print(json.dumps({k:index[k] for k in ("counts", "enabled_form", "form_support")}, indent=2))


if __name__ == "__main__":
    main()
