"""Evaluate both pipelines on identical reviewed test IDs, including abstentions."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from prototype.pipelines import MLPPipeline, YOLOPipeline
from prototype.schema import EXERCISES, FORMS, assert_no_leakage, form_support, path_under, read_manifest, save_json, valid_rows

ROOT = Path(__file__).resolve().parents[1]


def scores(rows, predictions):
    test_ids = {r["sample_id"] for r in rows}
    if set(predictions) != test_ids:
        raise ValueError("Different frame denominator")
    exercise_matrix = defaultdict(Counter)
    form_matrix = defaultdict(lambda: defaultdict(Counter))
    statuses = Counter()
    for row in rows:
        pred = predictions[row["sample_id"]]
        statuses[pred["status"]] += 1
        actual_ex = row["exercise"]
        exercise_matrix[actual_ex][pred["exercise"] or "abstain"] += 1
        if row["form"] in FORMS:
            output = pred["form"] if pred["status"] == "ok" else "abstain"
            if pred["exercise"] != actual_ex and pred["exercise"] is not None:
                output = "wrong_exercise"
            form_matrix[actual_ex][row["form"]][output] += 1
    reportable = form_support(rows, split="test")
    per_class = {}
    for exercise in EXERCISES:
        subset = [r for r in rows if r["exercise"] == exercise and r["form"] in FORMS]
        correct = sum(predictions[r["sample_id"]]["form"] == r["form"] and predictions[r["sample_id"]]["status"] == "ok"
                      and predictions[r["sample_id"]]["exercise"] == exercise for r in subset)
        gate = reportable[exercise]["reportable"]
        per_class[exercise] = {"frames":len(subset), "conditions":reportable[exercise]["conditions"],
                               "reportable":gate, "exploratory_accuracy":correct/len(subset) if subset and gate else None,
                               "confusion":{k:dict(v) for k,v in form_matrix[exercise].items()}}
    return {"denominator_frame_ids":sorted(test_ids), "frames":len(rows),
            "exercise_accuracy_exploratory":sum(predictions[r["sample_id"]]["exercise"] == r["exercise"] and
                                                 predictions[r["sample_id"]]["status"] not in ("no_person","multiple_people","uncertain")
                                                 for r in rows)/len(rows) if rows else None,
            "exercise_confusion":{k:dict(v) for k,v in exercise_matrix.items()},
            "statuses":dict(statuses), "abstention_rate":sum(v for k,v in statuses.items() if k != "ok")/len(rows) if rows else None,
            "form_per_class":per_class, "label":"exploratory"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    parser.add_argument("--pose-task", default="artifacts/pose_landmarker.task")
    parser.add_argument("--box-map", action="store_true", help="Run separate YOLO bbox mAP validation")
    args = parser.parse_args()
    rows = read_manifest(path_under(ROOT, args.manifest))
    valid_rows(rows, ROOT)
    assert_no_leakage(rows)
    test_rows = [r for r in rows if r["review_state"] == "accepted" and r["split"] == "test"]
    if not test_rows:
        raise SystemExit("No reviewed test frames")
    models = {"mlp":MLPPipeline(ROOT/"artifacts/mlp.joblib", path_under(ROOT,args.pose_task)),
              "yolo":YOLOPipeline(ROOT/"artifacts/yolo/exercise/weights/best.pt", ROOT/"artifacts/yolo/form/weights/best.pt")}
    predictions = {name:{} for name in models}
    latencies = {name:[] for name in models}
    for row in test_rows:
        image = cv2.imread(str(path_under(ROOT,row["image_path"])))
        for name, model in models.items():
            start = time.perf_counter()
            predictions[name][row["sample_id"]] = model.predict(image)
            latencies[name].append((time.perf_counter()-start)*1000)
    for model in models.values():
        model.close()
    ids = [{r["sample_id"] for r in test_rows} for _ in predictions]
    assert all(group == ids[0] for group in ids)
    report = {name:scores(test_rows, values) for name,values in predictions.items()}
    report["shared_denominator"] = sorted(ids[0])
    for name, values in latencies.items():
        report[name]["latency_ms_median"] = float(np.median(values))
        report[name]["latency_ms_p95"] = float(np.percentile(values,95))
    if args.box_map:
        from ultralytics import YOLO
        model = YOLO(str(ROOT/"artifacts/yolo/exercise/weights/best.pt"))
        metrics = model.val(data=str(ROOT/"data/derived/training/yolo_exercise/data.yaml"), split="test", workers=0)
        report["yolo_box_map_separate"] = {"exercise":{"map50":float(metrics.box.map50), "map50_95":float(metrics.box.map)}}
        form_weights = ROOT/"artifacts/yolo/form/weights/best.pt"
        index_path = ROOT/"data/derived/training/training_index.json"
        form_test_ids = json.loads(index_path.read_text(encoding="utf-8")).get("form_ids_by_split", {}).get("test", []) if index_path.exists() else []
        if form_weights.exists() and form_test_ids:
            form_model = YOLO(str(form_weights))
            form_metrics = form_model.val(data=str(ROOT/"data/derived/training/yolo_form/data.yaml"), split="test", workers=0)
            report["yolo_box_map_separate"]["form"] = {"map50":float(form_metrics.box.map50),
                                                         "map50_95":float(form_metrics.box.map)}
    save_json(ROOT/"artifacts/evaluation_exploratory.json", report)
    print(json.dumps({"frames":len(test_rows), "mlp_status":report["mlp"]["statuses"],
                      "yolo_status":report["yolo"]["statuses"], "label":"exploratory"}, indent=2))


if __name__ == "__main__":
    main()
