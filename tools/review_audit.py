"""Apply conservative visual-review decisions to generated audit manifests.

Raw source files are never opened for writing. This pass records inspection and
known contradictions; it does not create ground-truth posture labels.
"""
import collections
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manifests"


def read(name):
    with (M / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (M / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


inventory = read("source_inventory.csv")
samples = read("sample_selection.csv")
exclusions = read("exclusions.csv")
source_status = {
    "Dataset Subject-wise": "PRIMARY_CANDIDATE",
    "hmdb51": "NEEDS_RELABELING",
    "Plank_detection": "SEED_ONLY",
    "Mountain_Climbers_Pose_Detection": "REJECT",
    "Squat_Dataset_Zenodo": "SUPPLEMENTARY",
    "HSiPu2": "REFERENCE_ONLY",
    "FitAQA": "REFERENCE_ONLY",
    "README.md": "REFERENCE_ONLY",
}
by_hash = collections.defaultdict(list)
for row in inventory:
    if row["sha256"]:
        by_hash[row["sha256"]].append(row)
conflict_hashes = {h for h, rr in by_hash.items() if len(rr) > 1 and len({(r["exercise"], r["source_label"]) for r in rr}) > 1}
dupe_hashes = {h for h, rr in by_hash.items() if len(rr) > 1}
plank_groups = collections.defaultdict(list)
for row in inventory:
    if row["source_name"] == "Plank_detection" and row["asset_type"] == "image":
        plank_groups[row["augmentation_parent"]].append(row)
plank_representatives = {sorted(rr, key=lambda r: (";" in r["source_label"], r["relative_path"]))[0]["relative_path"] for rr in plank_groups.values()}

situp_files = sorted((ROOT / "data/sources/hmdb51/hmdb51_org/situp").rglob("*.avi"))
situp_index = {p.name: i for i, p in enumerate(situp_files)}
keep = {3, 4, 5, 30, 31, 32, 44, 45, 46}
partial = set(range(0, 3)) | set(range(15, 19)) | set(range(25, 28)) | {65, 66} | set(range(78, 81))
bad_view = {59, 60, 61, 76, 77}
not_situp = set(range(12, 15)) | set(range(19, 25)) | set(range(53, 56)) | set(range(67, 76)) | set(range(81, 84)) | set(range(87, 97)) | set(range(100, 105))


def screen(index):
    if index in keep:
        return "KEEP_FOR_LABELING"
    if index in partial:
        return "REJECT_PARTIAL_BODY"
    if index in bad_view:
        return "REJECT_BAD_VIEW"
    if index in not_situp:
        return "REJECT_NOT_SITUP"
    return "AMBIGUOUS"


def add_exclusion(path, code, detail):
    exclusions.append(dict(path=path, reason_code=code, reason_detail=detail))


for row in inventory + samples:
    src = row["source_name"]
    path = row.get("relative_path", row.get("path", ""))
    row["source_status"] = source_status[src]
    row["review_basis"] = ""
    row["duplicate_component"] = "sha256:" + row["sha256"][:16] if row.get("sha256") in dupe_hashes else ""
    if src == "Dataset Subject-wise":
        if row["exercise"] in {"push_up", "squat"}:
            if row["source_label"] == "bad":
                row["candidate_primary_class"] = ""
            if row["view"] != "side" and row["selection_status"] != "REJECT":
                row["selection_status"] = "SEED_ONLY"
                row["selection_reason"] = "Other view; reserve for possible group-linked training support after semantics review"
            row["review_basis"] = "side midframes for available clips; five phases for 13 representative clips"
        if row.get("sha256") in conflict_hashes and row["exercise"] in {"push_up", "squat"}:
            row["selection_status"] = "REJECT"
            row["selection_reason"] = "Identical bytes under conflicting exercise/form labels"
            row["candidate_primary_class"] = ""
        elif row.get("sha256") in dupe_hashes and len(by_hash[row["sha256"]]) > 1 and path != by_hash[row["sha256"]][0]["relative_path"] and row["exercise"] in {"push_up", "squat"}:
            row["selection_status"] = "REJECT"
            row["selection_reason"] = "Exact duplicate of another source video; subject IDs must be linked"
    elif src == "hmdb51" and path.lower().endswith(".avi"):
        ex = "pushup" if "\\pushup\\" in path.lower() or "/pushup/" in path.lower() else "situp"
        row["exercise"] = ex
        row["source_label"] = "activity_only"
        row["candidate_primary_class"] = ""
        row["review_basis"] = "105 sit-up originals screened at midpoint; six at five phases; push-up visual contact sheets"
        row["hmdb_screen"] = screen(situp_index[Path(path).name]) if ex == "situp" else "AMBIGUOUS"
        if "hmdb51_sta" in path:
            row["selection_status"] = "REFERENCE_ONLY"
            row["selection_reason"] = "Stabilized derivative of corresponding original clip; not independent"
        elif ex == "situp":
            c = row["hmdb_screen"]
            row["selection_status"] = "KEEP_FOR_LABELING" if c == "KEEP_FOR_LABELING" else "REJECT" if c.startswith("REJECT") else "AMBIGUOUS"
            row["selection_reason"] = "Initial visual screen: " + c + "; human clip and pose review still required"
        else:
            row["selection_status"] = "AMBIGUOUS"
            row["selection_reason"] = "Push-up activity only; form, view and full-repetition review needed"
    elif src == "Mountain_Climbers_Pose_Detection" and path.lower().endswith(".jpg"):
        row["selection_status"] = "REJECT"
        row["mc_audit_category"] = "LABEL_CUE_CONTAMINATED"
        row["selection_reason"] = "Visible right/wrong instructional text or check/cross; two repeated video families"
        row["candidate_primary_class"] = ""
        row["review_basis"] = "all 310 images screened in seven contact sheets"
    elif src == "Plank_detection" and path.lower().endswith(".jpg"):
        row["review_basis"] = "all 120 images screened in three contact sheets; source boxes parsed"
        row["independent_parent_representative"] = "yes" if path in plank_representatives else "no"
        row["is_augmented"] = "publisher_variant" if row.get("publisher_split") == "train" else "unknown"
        if ";" in row["source_label"]:
            row["selection_status"] = "AMBIGUOUS"
            row["selection_reason"] = "Multiple source posture classes/people in composite image; target-person identity unresolved"
            row["candidate_primary_class"] = ""
        else:
            row["selection_status"] = "SEED_ONLY"
            row["selection_reason"] = "Single source box, but parent, label, view and box need human review"
    elif src == "Squat_Dataset_Zenodo" and path.lower().endswith(".jpg"):
        row["review_basis"] = "stratified visual sheets by source class; pHash scan of all images"
        row["selection_reason"] = "Supplementary candidate; repeated frame families, no known subject/session, boxes or adjudicated proposal label"

for row in inventory:
    if row["source_name"] == "Mountain_Climbers_Pose_Detection" and row["asset_type"] == "image":
        add_exclusion(row["relative_path"], "ANSWER_LEAKAGE", "Visible label-specific instructional text or check/cross in full-image contact-sheet scan")
    if row["source_name"] == "Plank_detection" and row["asset_type"] == "image" and ";" in row["source_label"]:
        add_exclusion(row["relative_path"], "AMBIGUOUS_POSTURE", "Composite image has boxes for multiple posture classes; no target person for paired comparison")
    if row["source_name"] == "Plank_detection" and row["asset_type"] == "image" and row["relative_path"] not in plank_representatives:
        add_exclusion(row["relative_path"], "AUGMENTED_DERIVATIVE", "Additional exported variant of the same filename parent; not an independent observation")
    if row["source_name"] == "hmdb51" and row["asset_type"] == "video":
        if "hmdb51_sta" in row["relative_path"]:
            add_exclusion(row["relative_path"], "AUGMENTED_DERIVATIVE", "Stabilized rendering of corresponding original source clip")
        elif row["exercise"] == "situp" and row["hmdb_screen"].startswith("REJECT"):
            code = {"REJECT_PARTIAL_BODY": "PARTIAL_BODY", "REJECT_BAD_VIEW": "WRONG_VIEW", "REJECT_NOT_SITUP": "WRONG_EXERCISE"}[row["hmdb_screen"]]
            add_exclusion(row["relative_path"], code, "Initial midpoint visual screen; can be revisited by full-clip reviewer")
    if row.get("sha256") in conflict_hashes and row["source_name"] == "Dataset Subject-wise" and row["exercise"] in {"push_up", "squat"}:
        add_exclusion(row["relative_path"], "LABEL_MISMATCH", "Same SHA-256 appears with conflicting source exercise/form labels")

seen = set()
exclusions = [r for r in exclusions if not ((r["path"], r["reason_code"]) in seen or seen.add((r["path"], r["reason_code"]))) ]
write("source_inventory.csv", inventory)
write("sample_selection.csv", samples)
write("exclusions.csv", exclusions)
print("Reviewed", len(inventory), "inventory rows and", len(samples), "samples;", len(exclusions), "exclusions")
print("HMDB sit-up screen", dict(collections.Counter(screen(i) for i in range(len(situp_files)))))
