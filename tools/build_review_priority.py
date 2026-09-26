"""Select a smaller, provenance-aware review queue without promoting source labels."""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from prototype.schema import read_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = read_manifest(ROOT/"manifests/prototype_samples.csv")
    selected = {}

    def add(chosen, reason):
        for row in chosen:
            selected[row["sample_id"]] = (row, reason)

    for exercise in ("push_up", "squat"):
        for split, n_groups in (("train", 4), ("val", 1), ("test", 4)):
            subset = [r for r in rows if r["exercise_hint"] == exercise and r["split"] == split]
            groups = defaultdict(list)
            for row in subset:
                groups[row["group_id"]].append(row)
            ranked = sorted(groups, key=lambda group: (
                len({r["form_hint"] for r in groups[group]}) < 2,
                hashlib.sha256(group.encode()).hexdigest()))
            for group in ranked[:n_groups]:
                add(groups[group], f"Mendeley {split}: provenance group {group}; source labels are hints")
    plank = [r for r in rows if r["exercise_hint"] == "plank" and r["split"] == "train"]
    good = [r for r in plank if r["form_hint"] == "Good-plank"]
    bad = [r for r in plank if r["form_hint"] != "Good-plank"]
    add(good[:18] + bad[:12], "Plank recognition seed only; origin is one unknown group")
    add([r for r in rows if r["exercise_hint"] in ("sit_up", "mountain_climber")],
        "Activity/seed candidate; form requires fresh review and provenance gate")
    path = ROOT/"manifests/prototype_review_priority.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("sample_id", "source_name", "split", "exercise_hint", "form_hint", "reason"))
        writer.writeheader()
        for row, reason in selected.values():
            writer.writerow({"sample_id":row["sample_id"], "source_name":row["source_name"],
                             "split":row["split"], "exercise_hint":row["exercise_hint"],
                             "form_hint":row["form_hint"], "reason":reason})
    print(f"Wrote {len(selected)} prioritized candidate IDs to {path}")


if __name__ == "__main__":
    main()
