"""Summarize generated manifests and list perceptual duplicate candidates."""
import collections
import csv
import itertools
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manifests"


def read(name):
    with (M / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows = read("source_inventory.csv")
samples = read("sample_selection.csv")
exclusions = read("exclusions.csv")
images = [r for r in rows if r["asset_type"] == "image"]
videos = [r for r in rows if r["asset_type"] == "video"]
print("SOURCES")
for s in sorted({r["source_name"] for r in rows}):
    rr = [r for r in rows if r["source_name"] == s]
    print(s, len(rr), round(sum(int(r["file_size_bytes"]) for r in rr)/2**20, 1), dict(collections.Counter(r["asset_type"] for r in rr)))
print("EXCLUSIONS", dict(collections.Counter(r["reason_code"] for r in exclusions)))
print("EXACT", [(r["path"], r["reason_detail"]) for r in exclusions if r["reason_code"] == "DUPLICATE"][:40])
print("SAMPLE_STATUS", dict(collections.Counter((r["source_name"], r["selection_status"]) for r in samples)))
for source in sorted({r["source_name"] for r in images}):
    rr = [r for r in images if r["source_name"] == source]
    print("IMAGE_LABEL", source, dict(collections.Counter((r["source_label"], r["publisher_split"]) for r in rr)))
    print("IMAGE_SIZE", source, dict(collections.Counter((r["width"], r["height"]) for r in rr)))
for source in sorted({r["source_name"] for r in videos}):
    rr = [r for r in videos if r["source_name"] == source]
    vals = [float(r["duration_s"]) for r in rr if r["duration_s"]]
    print("VIDEO_META", source, "n", len(rr), "res", dict(collections.Counter((r["width"], r["height"]) for r in rr)), "fps", dict(collections.Counter(r["fps"] for r in rr)), "duration_min_med_max", (min(vals), statistics.median(vals), max(vals)) if vals else None)

print("MENDELEY_SIDE")
for subj in sorted({r["subject_id"] for r in videos if r["source_name"] == "Dataset Subject-wise"}):
    vals = []
    for ex in ["push_up", "squat"]:
        for lab in ["good", "bad"]:
            vals.append(sum(r["source_name"] == "Dataset Subject-wise" and r["subject_id"] == subj and r["exercise"] == ex and r["source_label"] == lab and r["view"] == "side" for r in videos))
    print(subj, *vals)

plank = [r for r in images if r["source_name"] == "Plank_detection"]
parents = collections.defaultdict(list)
for r in plank:
    parents[r["augmentation_parent"]].append(r)
print("PLANK_ORIGINAL_GROUPS", len(parents), dict(collections.Counter((rr[0]["source_label"], len(rr)) for rr in parents.values())))
print("PLANK_MIXED_PARENT_CLASSES", [k for k, rr in parents.items() if len({x["source_label"] for x in rr}) > 1])

mc = [r for r in images if r["source_name"] == "Mountain_Climbers_Pose_Detection"]
print("MC_VIDEO_FAMILIES", dict(collections.Counter(r["video_id"] for r in mc)))
mcstems = collections.defaultdict(list)
for r in mc:
    mcstems[r["augmentation_parent"]].append(r)
print("MC_REUSED_STEMS", sum(len(rr) > 1 for rr in mcstems.values()), "CROSS_CLASS", sum(len({x["source_label"] for x in rr}) > 1 for rr in mcstems.values()), "CROSS_SPLIT", sum(len({x["publisher_split"] for x in rr}) > 1 for rr in mcstems.values()))

# A Hamming threshold of 4 on 64-bit pHash is a retrieval heuristic, not proof
# that two independent images are the same source observation.
candidates = []
for a, b in itertools.combinations(images, 2):
    if a["phash"] and b["phash"] and (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() <= 4:
        candidates.append(dict(path_a=a["relative_path"], path_b=b["relative_path"],
                               source_a=a["source_name"], source_b=b["source_name"],
                               label_a=a["source_label"], label_b=b["source_label"],
                               split_a=a["publisher_split"], split_b=b["publisher_split"],
                               phash_distance=(int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count()))
review_pairs = [x for x in candidates if x["phash_distance"] <= 1 and (x["label_a"] != x["label_b"] or x["split_a"] != x["split_b"] or x["source_a"] != x["source_b"])]
with (M / "near_duplicate_candidates.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(candidates[0]) if candidates else ["path_a", "path_b"])
    w.writeheader()
    w.writerows(review_pairs)
print("PHASH_PAIRS", len(candidates), "CROSS_SOURCE", sum(x["source_a"] != x["source_b"] for x in candidates), "CROSS_CLASS", sum(x["label_a"] != x["label_b"] for x in candidates), "CROSS_PUBLISHER_SPLIT", sum(x["split_a"] != x["split_b"] and x["source_a"] == x["source_b"] for x in candidates))
print("PHASH_BY_SOURCE", dict(collections.Counter((x["source_a"], x["source_b"]) for x in candidates)))
print("HIGH_PRIORITY_PHASH_PAIRS", len(review_pairs), dict(collections.Counter((x["source_a"], x["phash_distance"]) for x in review_pairs)))
