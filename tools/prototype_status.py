"""Read-only progress report for reviewer and training gates."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import argparse

from prototype.schema import assert_no_leakage, box, form_support, read_manifest, save_json, valid_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write manifests/prototype_readiness.json")
    args = parser.parse_args()
    rows = read_manifest(ROOT/"manifests/prototype_samples.csv")
    valid_rows(rows, ROOT)
    assert_no_leakage(rows)
    report = {"candidates":len(rows), "review_state":dict(Counter(r["review_state"] for r in rows)),
              "candidate_by_exercise":dict(Counter(r["exercise_hint"] for r in rows)),
              "proposed_boxes":sum(box(r) is not None for r in rows),
              "boxes_needing_manual_draw":dict(Counter(r["exercise_hint"] for r in rows if box(r) is None)),
              "reviewed_by_exercise":dict(Counter(r["exercise"] for r in rows if r["review_state"] == "accepted")),
              "form_train_gate":form_support(rows, "train"), "form_test_gate":form_support(rows, "test"),
              "models":{"mlp":(ROOT/"artifacts/mlp.joblib").exists(),
                        "yolo_exercise":(ROOT/"artifacts/yolo/exercise/weights/best.pt").exists(),
                        "yolo_form":(ROOT/"artifacts/yolo/form/weights/best.pt").exists()},
              "pose_task":(ROOT/"artifacts/pose_landmarker.task").exists()}
    if args.write:
        save_json(ROOT/"manifests/prototype_readiness.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
