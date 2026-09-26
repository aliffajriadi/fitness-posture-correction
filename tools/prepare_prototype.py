"""Create frame candidates; source labels remain hints and all rows stay pending.

Run from repo root: python -m tools.prepare_prototype
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import cv2

from prototype.pose import sha256_file
from prototype.schema import FIELDS, assign_splits, path_under, save_json, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def source_rows():
    with (ROOT / "manifests/sample_selection.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def proposed_box(source, width, height):
    label_path = source.with_name(source.name.replace("images", "labels"))
    label_path = source.parent.parent / "labels" / (source.stem + ".txt")
    if not label_path.exists():
        return None
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 1:
        return None
    parts = lines[0].split()
    if len(parts) != 5:
        return None
    _, xc, yc, w, h = map(float, parts)
    return (max(0, round((xc-w/2)*width)), max(0, round((yc-h/2)*height)),
            min(width, round((xc+w/2)*width)), min(height, round((yc+h/2)*height)))


def video_frames(source, count=3):
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total <= 0:
        cap.release()
        return
    # Three temporal bins; pick the largest visible change within each bin.
    centers = (.15, .325, .50, .675, .85) if count == 5 else (.25, .50, .75)
    for center in centers:
        indices = sorted({max(0, min(total-1, round(total*(center+delta)))) for delta in (-.045, -.02, 0, .02, .045)})
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if ok:
                frames.append((index, image))
        if not frames:
            continue
        scored = []
        for n, (index, image) in enumerate(frames):
            if n == 0:
                motion = 0.0
            else:
                before = cv2.cvtColor(cv2.resize(frames[n-1][1], (160, 90)), cv2.COLOR_BGR2GRAY)
                after = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY)
                motion = float(cv2.absdiff(before, after).mean())
            scored.append((motion, index, image))
        _, index, image = max(scored, key=lambda item: item[0])
        yield index, index / fps if fps else 0, image
    cap.release()


def mc_subset(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["video_id"], row["source_label"])].append(row)
    selected = []
    for group in by.values():
        group.sort(key=lambda x: x["path"])
        for n in range(min(10, len(group))):
            selected.append(group[round(n*(len(group)-1)/max(1, min(10, len(group))-1))])
    return selected


def make_row(asset, image, index, time_s, crop=None, proposed=None):
    source_path = asset["path"].replace("\\", "/")
    token = f"{source_path}:{index}:{crop}"
    sid = hashlib.sha256(token.encode()).hexdigest()[:20]
    h, w = image.shape[:2]
    source = path_under(ROOT, source_path)
    if crop:
        x1, y1, x2, y2 = crop
        image = image[y1:y2, x1:x2]
        h, w = image.shape[:2]
        proposed = (0, 0, w, h)
    destination = ROOT / "data/derived/frames" / f"{sid}.jpg"
    if not destination.exists() and not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Cannot write {destination}")
    row = dict.fromkeys(FIELDS, "")
    row.update(sample_id=sid, source_path=source_path,
               image_path=destination.relative_to(ROOT).as_posix(), image_sha256=sha256_file(destination),
               source_name=asset["source_name"],
               source_sha256=asset["sha256"], parent_id=asset["augmentation_parent"] or asset["video_id"] or asset["sample_id"],
               group_id=("Plank_detection:unknown_subject_session" if asset["source_name"] == "Plank_detection" else
                         asset["group_id"] or f"{asset['source_name']}:{asset['video_id'] or asset['sample_id']}"),
               duplicate_component=asset["duplicate_component"], frame_index=str(index), time_s=f"{time_s:.3f}",
               width=str(w), height=str(h), exercise_hint={"situp":"sit_up", "pushup":"push_up"}.get(asset["exercise"], asset["exercise"]),
               form_hint=asset["source_label"], phase="unknown", uncertain="yes", review_state="pending",
               bbox_reviewed="no", cue_free="no", geometry_validated="no")
    if crop:
        for key, value in zip(("crop_x1", "crop_y1", "crop_x2", "crop_y2"), crop):
            row[key] = str(value)
    if proposed:
        for key, value in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), proposed):
            row[key] = str(value)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    parser.add_argument("--replace-pending", action="store_true", help="Replace a manifest only if every row is still pending")
    args = parser.parse_args()
    out = path_under(ROOT, args.manifest)
    if out.exists():
        if not args.replace_pending:
            raise SystemExit(f"Manifest exists: {out}. Existing human reviews must not be overwritten.")
        from prototype.schema import read_manifest
        if any(r["review_state"] != "pending" for r in read_manifest(out)):
            raise SystemExit("Refusing to overwrite human review decisions")
    (ROOT / "data/derived/frames").mkdir(parents=True, exist_ok=True)
    source = source_rows()
    selected = [r for r in source if r["source_name"] == "Dataset Subject-wise" and r["exercise"] in ("push_up", "squat")
                and r["view"] == "side" and r["selection_status"] == "KEEP_FOR_LABELING"]
    selected += [r for r in source if r["source_name"] == "Plank_detection" and r["selection_status"] == "SEED_ONLY"
                 and r["independent_parent_representative"] == "yes"]
    selected += [r for r in source if r["source_name"] == "hmdb51" and r["exercise"] == "situp"
                 and r["selection_status"] == "KEEP_FOR_LABELING" and "hmdb51_org" in r["path"]]
    selected += mc_subset([r for r in source if r["source_name"] == "Mountain_Climbers_Pose_Detection" and r["path"].lower().endswith(".jpg")])
    rows = []
    failures = []
    for n, asset in enumerate(selected, 1):
        path = path_under(ROOT, asset["path"])
        try:
            if path.suffix.lower() in (".avi", ".mp4", ".mov"):
                for index, time_s, image in video_frames(path, 5 if asset["source_name"] == "Dataset Subject-wise" else 3):
                    rows.append(make_row(asset, image, index, time_s))
            else:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("Cannot decode image")
                h, w = image.shape[:2]
                proposed = proposed_box(path, w, h)
                if asset["source_name"] == "Plank_detection" and proposed is None:
                    raise ValueError("Plank candidate has zero or multiple source person boxes")
                crop = proposed if asset["source_name"] == "Mountain_Climbers_Pose_Detection" else None
                if crop and (crop[2]-crop[0] < 32 or crop[3]-crop[1] < 32):
                    raise ValueError("Source box too small")
                rows.append(make_row(asset, image, 0, 0, crop=crop, proposed=proposed))
        except Exception as error:
            failures.append({"source_path":asset["path"], "error":str(error)})
        if n % 25 == 0:
            print(f"Prepared {n}/{len(selected)} assets", flush=True)
    assign_splits(rows)
    write_manifest(out, rows)
    save_json(ROOT / "manifests/prototype_preparation.json", {"candidates": len(rows), "source_assets": len(selected),
                                                            "pending_review": len(rows), "failures": failures,
                                                            "selection": "Temporal motion candidates; phase and form are unreviewed."})
    print(f"Wrote {len(rows)} pending samples to {out}; {len(failures)} failures")


if __name__ == "__main__":
    main()
