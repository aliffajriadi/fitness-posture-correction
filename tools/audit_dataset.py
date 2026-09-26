"""Read-only raw-data inventory and audit manifest generator.

Run from the repository root. Only writes manifests/ and manifests/previews/.
No label here is treated as adjudicated ground truth.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources"
OUT = ROOT / "manifests"
PREVIEWS = OUT / "previews"
OUT.mkdir(exist_ok=True)
PREVIEWS.mkdir(exist_ok=True)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(image):
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8]
    bits = (dct > np.median(dct[1:, :])).flatten()
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"


def video_meta(path, sample_frame=True):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=round(fps, 3) if fps > 0 else None,
                frame_count=count,
                duration_s=round(count / fps, 3) if fps > 0 else None)
    frame = None
    if count and sample_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
        ok, frame = cap.read()
        if not ok:
            frame = None
    cap.release()
    return meta, frame


def source_of(rel):
    return rel.parts[2] if len(rel.parts) > 2 else "unknown"


def classify(rel):
    p = rel.parts
    src = source_of(rel)
    name = rel.stem
    d = dict(exercise="", source_label="", view="", subject_id="",
             video_id="", frame_id="", is_augmented="", augmentation_parent="",
             group_id="", candidate_primary_class="", error_subtype="",
             selection_status="REFERENCE_ONLY", selection_reason="Nonvisual reference/metadata",
             proposal_compatible="", bbox_available="no")
    if src == "Dataset Subject-wise" and len(p) >= 8:
        subj, exercise, label, view = p[3:7]
        d.update(exercise=exercise, source_label=label, view=view, subject_id=subj,
                 video_id=name, group_id=f"mendeley:{subj}",
                 selection_status="KEEP_FOR_LABELING" if exercise in {"push_up", "squat"} else "REFERENCE_ONLY",
                 selection_reason="Subject/view known; frame-level proposal semantics and boxes need review" if exercise in {"push_up", "squat"} else "Exercise outside target five",
                 proposal_compatible="pending_semantic_review" if exercise in {"push_up", "squat"} else "no")
        if exercise in {"push_up", "squat"}:
            d["candidate_primary_class"] = ("pushup" if exercise == "push_up" else "squat") + ("_correct" if label == "good" else "_incorrect")
    elif src == "hmdb51" and rel.suffix.lower() == ".avi":
        d.update(exercise="situp", source_label="activity_only", video_id=name,
                 group_id="hmdb:" + re.sub(r"_\d+$", "", name),
                 selection_status="AMBIGUOUS" if "hmdb51_sta" in p else "KEEP_FOR_LABELING",
                 selection_reason="Activity-only; form/view/full body require clip review" if "hmdb51_org" in p else "Alternate representation of same source clip; do not count independently",
                 proposal_compatible="pending_review")
    elif src in {"Mountain_Climbers_Pose_Detection", "Plank_detection"} and rel.suffix.lower() in IMAGE_EXT:
        stem = name.split("_jpg.rf.")[0]
        d.update(exercise="mountain_climber" if src.startswith("Mountain") else "plank",
                 video_id=stem.split("_frame")[0] if "_frame" in stem else "",
                 frame_id=stem.split("_frame")[-1] if "_frame" in stem else "",
                 augmentation_parent=stem, group_id=src + ":" + (stem.split("_frame")[0] if src.startswith("Mountain") else stem),
                 is_augmented="unknown" if src.startswith("Mountain") else str(p[3] == "train" and stem in plank_multi),
                 bbox_available="yes_unverified", selection_status="SEED_ONLY",
                 selection_reason="Publisher label/box and independence need visual review",
                 proposal_compatible="pending_review")
    elif src == "Squat_Dataset_Zenodo" and rel.suffix.lower() in IMAGE_EXT:
        lab = p[4].lower() if len(p) > 4 else ""
        d.update(exercise="squat", source_label=lab, group_id="zenodo:unknown:" + name,
                 selection_status="KEEP_FOR_LABELING", selection_reason="Source class candidate; person/session and proposal semantics unverified; no boxes",
                 proposal_compatible="pending_review")
        if lab == "good":
            d["candidate_primary_class"] = "squat_correct"
        elif lab in {"bad back", "bad heel"}:
            d["candidate_primary_class"] = "squat_incorrect"
            d["error_subtype"] = lab.replace(" ", "_")
    elif src == "HSiPu2" and rel.suffix.lower() == ".csv":
        d.update(source_label="standard" if "standard" in p and "not_standard" not in p else "not_standard",
                 selection_reason="17-keypoint CSV; no paired RGB pixels locally", proposal_compatible="no")
    return d


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sheet(name, items, columns=6, thumb=(192, 144)):
    if not items:
        return
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + 38)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(items):
        x, y = i % columns * thumb[0], i // columns * (thumb[1] + 38)
        canvas.paste(ImageOps.contain(im.convert("RGB"), thumb), (x, y))
        draw.text((x + 2, y + thumb[1] + 2), label[:31], fill="black")
        if len(label) > 31:
            draw.text((x + 2, y + thumb[1] + 18), label[31:62], fill="black")
    canvas.save(PREVIEWS / name, quality=85)


all_files = sorted(x for x in SOURCES.rglob("*") if x.is_file() and ".git" not in x.parts)
plank_stems = collections.Counter(x.stem.split("_jpg.rf.")[0] for x in all_files
                                   if "Plank_detection" in x.parts and x.suffix.lower() in IMAGE_EXT)
plank_multi = {k for k, v in plank_stems.items() if v > 1}
inventory, samples, exclusions = [], [], []
digest_map = collections.defaultdict(list)
image_hashes = []
source_stats = collections.defaultdict(lambda: collections.Counter())
image_previews = collections.defaultdict(list)
video_previews = collections.defaultdict(list)
bad_boxes = []

for n, path in enumerate(all_files, 1):
    rel = path.relative_to(ROOT)
    src = source_of(rel)
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        typ = "image"
    elif ext in VIDEO_EXT:
        typ = "video"
    elif ext == ".csv":
        typ = "keypoint_csv"
    elif ext == ".parquet":
        typ = "annotation_parquet"
    else:
        typ = "reference_or_archive"
    d = classify(rel)
    if src in {"Plank_detection", "Mountain_Climbers_Pose_Detection"} and typ == "image":
        labels = path.parent.parent / "labels" / (path.stem + ".txt")
        if labels.exists():
            vals = []
            for line in labels.read_text(encoding="utf-8").splitlines():
                try:
                    v = [float(x) for x in line.split()]
                    if len(v) != 5 or v[0] not in range(2 if src.startswith("Mountain") else 3) or not all(0 <= x <= 1 for x in v[1:]) or v[3] <= 0 or v[4] <= 0 or v[1]-v[3]/2 < 0 or v[1]+v[3]/2 > 1 or v[2]-v[4]/2 < 0 or v[2]+v[4]/2 > 1:
                        bad_boxes.append(str(rel))
                    vals.append(v)
                except ValueError:
                    bad_boxes.append(str(rel))
            if vals:
                ids = sorted({int(v[0]) for v in vals if len(v) == 5})
                names = ["correct", "incorrect"] if src.startswith("Mountain") else ["Good-plank", "Too-high-plank", "Too-low-plank"]
                d["source_label"] = ";".join(names[i] for i in ids if i < len(names))
                if len(ids) == 1:
                    i = ids[0]
                    if src.startswith("Mountain"):
                        d["candidate_primary_class"] = "mountain_climber_" + names[i]
                    else:
                        d["candidate_primary_class"] = "plank_correct" if i == 0 else "plank_incorrect"
                        d["error_subtype"] = "" if i == 0 else ("hip_too_high" if i == 1 else "hip_too_low")
                else:
                    d["selection_status"] = "AMBIGUOUS"
                    d["selection_reason"] = "Multiple source classes in one image; target person unresolved"
        else:
            bad_boxes.append(str(rel))
            d["bbox_available"] = "no"
    meta = {}
    im = None
    try:
        if typ == "image":
            with Image.open(path) as src_im:
                im = src_im.convert("RGB")
                meta = dict(width=im.width, height=im.height)
                image_hashes.append((str(rel), src, phash(im), d["source_label"], d["augmentation_parent"]))
        elif typ == "video":
            inspect_frame = (src == "Dataset Subject-wise" and d["exercise"] in {"push_up", "squat"} and d["view"] == "side") or (src == "hmdb51" and "hmdb51_org" in rel.parts)
            meta, frame = video_meta(path, inspect_frame)
            if frame is not None:
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        d["selection_status"] = "REJECT"
        d["selection_reason"] = "Decode failed: " + str(e)[:100]
        exclusions.append(dict(path=str(rel), reason_code="LOW_QUALITY", reason_detail=d["selection_reason"]))
    digest = sha256(path) if typ in {"image", "video", "keypoint_csv", "annotation_parquet"} else ""
    if digest:
        digest_map[digest].append(str(rel))
    row = dict(source_name=src, source_path=str(path.resolve()), relative_path=str(rel), asset_type=typ,
               file_size_bytes=path.stat().st_size, sha256=digest, phash=image_hashes[-1][2] if typ == "image" and image_hashes and image_hashes[-1][0] == str(rel) else "",
               publisher_split=next((x for x in rel.parts if x in {"train", "valid", "test"}), ""),
               annotation_format="YOLO normalized xywh" if d["bbox_available"] == "yes_unverified" else "",
               rgb_available=str(typ in {"image", "video"}), **meta, **d)
    inventory.append(row)
    source_stats[src]["files"] += 1
    source_stats[src]["bytes"] += path.stat().st_size
    source_stats[src][typ] += 1
    if typ in {"image", "video", "keypoint_csv", "annotation_parquet"}:
        samples.append(dict(sample_id=hashlib.sha1(str(rel).encode()).hexdigest()[:16],
                            source_name=src, path=str(rel), **d, **meta,
                            sha256=digest, publisher_split=row["publisher_split"]))
    if im is not None:
        if typ == "image" and len(image_previews[src]) < 48:
            image_previews[src].append((im.copy(), f"{d['source_label']} {path.stem[:22]}"))
        if typ == "video" and src == "Dataset Subject-wise" and d["exercise"] in {"push_up", "squat"} and d["view"] == "side" and len(video_previews[d["exercise"]]) < 48:
            video_previews[d["exercise"]].append((im.copy(), f"{d['subject_id']} {d['source_label']}"))
        if typ == "video" and src == "hmdb51" and "hmdb51_org" in rel.parts and len(video_previews["situp"]) < 48:
            video_previews["situp"].append((im.copy(), path.stem[:30]))
    if n % 100 == 0:
        print("processed", n, "of", len(all_files), flush=True)

for digest, paths in digest_map.items():
    if len(paths) > 1:
        for dup in paths[1:]:
            exclusions.append(dict(path=dup, reason_code="DUPLICATE", reason_detail=f"Exact SHA-256 duplicate of {paths[0]}"))
            for s in samples:
                if s["path"] == dup and s["selection_status"] != "REFERENCE_ONLY":
                    s["selection_status"] = "REJECT"
                    s["selection_reason"] = "Exact duplicate; see exclusions.csv"
for rel in sorted(set(bad_boxes)):
    exclusions.append(dict(path=rel, reason_code="INVALID_ANNOTATION", reason_detail="Missing, malformed or out-of-bounds YOLO box"))

write_csv("source_inventory.csv", inventory, list(dict.fromkeys(k for row in inventory for k in row)))
write_csv("sample_selection.csv", samples, list(dict.fromkeys(k for row in samples for k in row)))
write_csv("exclusions.csv", exclusions, ["path", "reason_code", "reason_detail"])
for src, items in image_previews.items():
    sheet(src + ".jpg", items)
for ex, items in video_previews.items():
    sheet("video_" + ex + ".jpg", items)

summary = dict(total_files=len(all_files), sources={k: dict(v) for k, v in source_stats.items()},
               exact_duplicate_sets=[v for v in digest_map.values() if len(v) > 1],
               invalid_box_images=sorted(set(bad_boxes)),
               plank_original_stems=len(plank_stems),
               plank_derivative_stems=sum(v > 1 for v in plank_stems.values()),
               counts_by_source_label=dict(collections.Counter((r["source_name"], r["exercise"], r["source_label"], r["view"]) for r in samples)),
               video_metadata={"mendeley_side": {ex: [dict(path=r["path"], subject=r["subject_id"], label=r["source_label"], width=r.get("width"), height=r.get("height"), fps=r.get("fps"), duration_s=r.get("duration_s")) for r in samples if r["source_name"] == "Dataset Subject-wise" and r["exercise"] == ex and r["view"] == "side"] for ex in ["push_up", "squat"]}})
# Tuple keys need a list representation for JSON.
summary["counts_by_source_label"] = [dict(source=k[0], exercise=k[1], label=k[2], view=k[3], count=v) for k, v in summary["counts_by_source_label"].items()]
(OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("done", len(inventory), "inventory rows;", len(samples), "sample rows;", len(exclusions), "exclusions", flush=True)
