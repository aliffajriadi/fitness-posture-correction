"""Shared labels, manifest validation, split rules, and prediction contract."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXERCISES = ("plank", "mountain_climber", "push_up", "sit_up", "squat")
FORMS = ("correct", "incorrect")
FORM_LABELS = tuple(f"{exercise}_{form}" for exercise in EXERCISES for form in FORMS)
PHASES = ("static", "top", "middle", "bottom", "transition", "unknown")
STATUSES = ("ok", "no_person", "multiple_people", "uncertain", "waiting_phase", "unsupported_form")
RESULT_FIELDS = ("exercise", "form", "status", "confidence", "bbox", "feedback")
FIELDS = (
    "sample_id", "source_path", "image_path", "image_sha256", "source_name", "source_sha256",
    "parent_id", "group_id", "duplicate_component", "frame_index", "time_s",
    "width", "height", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    "exercise_hint", "form_hint", "exercise", "form", "error_type", "phase",
    "uncertain", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "review_state", "bbox_reviewed", "cue_free", "geometry_validated", "reviewer", "review_note", "split",
)


def encode_form_label(exercise, form):
    if exercise not in EXERCISES or form not in FORMS:
        raise ValueError((exercise, form))
    return f"{exercise}_{form}"


def decode_form_label(label):
    if label not in FORM_LABELS:
        raise ValueError(label)
    exercise, form = label.rsplit("_", 1)
    return exercise, form


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        return list(reader)


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def path_under(root, relative):
    path = (Path(root) / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes workspace: {relative}")
    return path


def box(row):
    values = [row.get("bbox_" + k, "") for k in ("x1", "y1", "x2", "y2")]
    if not all(v != "" for v in values):
        return None
    x1, y1, x2, y2 = (int(float(v)) for v in values)
    if not (0 <= x1 < x2 <= int(row["width"]) and 0 <= y1 < y2 <= int(row["height"])):
        raise ValueError(f"Invalid bbox for {row['sample_id']}: {values}")
    return (x1, y1, x2, y2)


def valid_rows(rows, root=None):
    seen = set()
    for row in rows:
        sid = row["sample_id"]
        if not sid or sid in seen:
            raise ValueError(f"Missing or repeated sample_id: {sid}")
        seen.add(sid)
        if row["review_state"] not in ("pending", "accepted", "rejected"):
            raise ValueError(f"Invalid review_state: {sid}")
        if row["exercise"] and row["exercise"] not in EXERCISES:
            raise ValueError(f"Invalid exercise: {sid}")
        if row["form"] and row["form"] not in FORMS:
            raise ValueError(f"Invalid form: {sid}")
        if row["phase"] and row["phase"] not in PHASES:
            raise ValueError(f"Invalid phase: {sid}")
        if row["split"] and row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {sid}")
        if not row["group_id"] or not row["parent_id"]:
            raise ValueError(f"Missing provenance group: {sid}")
        if row["form"] == "correct" and row["error_type"]:
            raise ValueError(f"Correct form has error_type: {sid}")
        if row["review_state"] == "accepted":
            if not row["exercise"] or row["uncertain"] != "no" or row["cue_free"] != "yes":
                raise ValueError(f"Accepted row lacks exercise or certainty/cue check: {sid}")
            if not row["reviewer"] or not row["phase"] or row["phase"] == "unknown":
                raise ValueError(f"Accepted row lacks reviewer or phase: {sid}")
            if row["form"] == "incorrect" and not row["error_type"]:
                raise ValueError(f"Incorrect row lacks error_type: {sid}")
            if box(row) is None or row["bbox_reviewed"] != "yes":
                raise ValueError(f"Accepted row lacks reviewed bbox: {sid}")
        else:
            box(row) if all(row.get("bbox_" + k, "") for k in ("x1", "y1", "x2", "y2")) else None
        if root is not None:
            if not path_under(root, row["source_path"]).is_file():
                raise ValueError(f"Missing source: {sid}")
            image_path = path_under(root, row["image_path"])
            if not image_path.is_file():
                raise ValueError(f"Missing derived image: {sid}")
            if row["review_state"] == "accepted":
                from .pose import sha256_file
                if not row["image_sha256"] or sha256_file(image_path) != row["image_sha256"]:
                    raise ValueError(f"Reviewed image changed after annotation: {sid}")
    return True


def _group_components(rows):
    """Union subject/recording/augmentation/duplicate links; one split per component."""
    parent = {r["sample_id"]: r["sample_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    owners = {}
    for row in rows:
        sid = row["sample_id"]
        for field in ("group_id", "parent_id", "source_sha256", "duplicate_component"):
            value = row.get(field, "")
            if not value:
                continue
            key = (field, value)
            if key in owners:
                union(sid, owners[key])
            else:
                owners[key] = sid
    components = defaultdict(list)
    for row in rows:
        components[find(row["sample_id"])].append(row)
    return list(components.values())


def assign_splits(rows):
    """Deterministic source-stratified provenance split shared by both pipelines.

    Small sources receive train/test only; a single unknown-origin group is train
    seed only and cannot supply independent validation or test evidence.
    """
    components = defaultdict(list)
    for component in _group_components(rows):
        source = "|".join(sorted({r["source_name"] for r in component}))
        components[source].append(component)
    for source, groups in components.items():
        groups.sort(key=lambda rr: hashlib.sha256((source + "|" + "|".join(sorted({r["group_id"] for r in rr}))).encode()).hexdigest())
        n = len(groups)
        test_n = max(3, round(.20*n)) if n >= 10 else (1 if n >= 2 else 0)
        val_n = max(1, round(.15*n)) if n >= 6 else 0
        for i, component in enumerate(groups):
            split = "test" if i < test_n else "val" if i < test_n + val_n else "train"
            for row in component:
                row["split"] = split
    assert_no_leakage(rows)
    return rows


def assert_no_leakage(rows):
    for component in _group_components(rows):
        splits = {r["split"] for r in component if r["split"]}
        if len(splits) > 1:
            raise ValueError(f"Provenance leakage across {splits}: {[r['sample_id'] for r in component[:4]]}")


def component_ids(rows):
    """Stable ID per independent provenance component (including duplicate links)."""
    return {r["sample_id"]:min(x["sample_id"] for x in component)
            for component in _group_components(rows) for r in component}


def form_support(rows, split="train"):
    """Conservative per-exercise gate. Unknown provenance is one group, never many."""
    result = {}
    component_id = component_ids(rows)
    for exercise in EXERCISES:
        by_form = {}
        for form in FORMS:
            subset = [r for r in rows if r["review_state"] == "accepted" and r["split"] == split
                      and r["exercise"] == exercise and r["form"] == form]
            groups = {component_id[r["sample_id"]] for r in subset}
            by_form[form] = {"frames": len(subset), "groups": len(groups)}
        enabled = all(v["frames"] >= 10 and v["groups"] >= 2 for v in by_form.values())
        reportable = all(v["frames"] >= 15 and v["groups"] >= 3 for v in by_form.values())
        result[exercise] = {"conditions": by_form, "enabled" if split == "train" else "reportable": enabled if split == "train" else reportable}
    return result


def prediction(exercise=None, form=None, status="uncertain", confidence=0.0, bbox=None, feedback=""):
    if exercise is not None and exercise not in EXERCISES:
        raise ValueError(exercise)
    if form is not None and form not in FORMS:
        raise ValueError(form)
    if status not in STATUSES:
        raise ValueError(status)
    if status != "ok":
        form = None
    return dict(exercise=exercise, form=form, status=status,
                confidence=float(confidence), bbox=list(bbox) if bbox is not None else None,
                feedback=feedback)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
