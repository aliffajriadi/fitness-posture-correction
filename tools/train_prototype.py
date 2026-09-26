"""Train both independent branches only after explicit audit approval.

Example: python -m tools.train_prototype --audit-approved-by "Reviewer Name" --branch mlp
The approval name is recorded against the exact reviewed manifest hash.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from prototype.pose import sha256_file
from prototype.schema import EXERCISES, FORMS, form_support, path_under, read_manifest, save_json, valid_rows

ROOT = Path(__file__).resolve().parents[1]


def run_mlp(rows, index, out):
    import joblib
    import sklearn
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cache = np.load(ROOT / "data/derived/training/features.npz")
    features = dict(zip(cache["ids"].tolist(), cache["features"]))
    lookup = {r["sample_id"]: r for r in rows}
    ids = index["recognition_train_ids"]
    counts = Counter(lookup[i]["exercise"] for i in ids)
    if set(counts) != set(EXERCISES) or any(counts[ex] < 10 for ex in EXERCISES):
        raise SystemExit(f"Five-class MLP requires >=10 pose-readable reviewed train frames per exercise: {counts}")
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=350,
                                                          random_state=42, early_stopping=False))
    model.fit(np.stack([features[i] for i in ids]), [lookup[i]["exercise"] for i in ids])
    form_models = {}
    for exercise in index["enabled_form"]:
        train_ids = [i for i in index["form_training_ids"] if lookup[i]["exercise"] == exercise]
        sub = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350,
                                                            random_state=42, early_stopping=False))
        sub.fit(np.stack([features[i] for i in train_ids]), [lookup[i]["form"] for i in train_ids])
        form_models[exercise] = sub
    out.mkdir(parents=True, exist_ok=True)
    target = out / "mlp.joblib"
    joblib.dump({"exercise":model, "form":form_models}, target)
    save_json(out / "mlp_metadata.json", {"feature_spec":index["feature_spec"], "manifest_sha256":index["manifest_sha256"],
                                         "pose_model_sha256":index["pose_model_sha256"], "train_ids":ids,
                                         "form_training_ids":index["form_training_ids"], "enabled_form":index["enabled_form"],
                                         "reviewed_phases":{ex:sorted({lookup[i]["phase"] for i in index["form_training_ids"] if lookup[i]["exercise"]==ex}) for ex in index["enabled_form"]},
                                         "python":platform.python_version(), "scikit_learn":sklearn.__version__,
                                         "model_sha256":sha256_file(target)})


def run_yolo(rows, index, out, epochs, device, batch, imgsz):
    config = ROOT / "artifacts/ultralytics_config"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
    import ultralytics
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    local_pretrained = out / "yolo12n_pretrained.pt"
    if not local_pretrained.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        shutil.copyfile(Path(attempt_download_asset("yolo12n.pt")), local_pretrained)
    if device == "auto":
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    kwargs = dict(epochs=epochs, imgsz=imgsz, batch=batch, workers=0, device=device, seed=42,
                  project=str(out), exist_ok=True, pretrained=True, verbose=True)
    results = {}
    for name in ("exercise", "form"):
        if name == "form" and not index["enabled_form"]:
            continue
        data_path = ROOT / "data/derived/training" / f"yolo_{name}/data.yaml"
        model = YOLO(str(local_pretrained))
        model.train(data=str(data_path), name=name, **kwargs)
        best = out / name / "weights/best.pt"
        results[name] = {"weights":str(best.relative_to(ROOT)), "sha256":sha256_file(best)}
    save_json(out / "yolo_metadata.json", {"manifest_sha256":index["manifest_sha256"],
                                          "pretrained_yolo12n_sha256":sha256_file(local_pretrained),
                                          "pretrained_yolo12n_path":str(local_pretrained.relative_to(ROOT)), "ultralytics":ultralytics.__version__,
                                          "python":platform.python_version(), "train_ids":index["recognition_train_ids"],
                                          "form_training_ids":index["form_training_ids"], "enabled_form":index["enabled_form"],
                                          "reviewed_phases":{ex:sorted({r["phase"] for r in rows if r["sample_id"] in index["form_training_ids"] and r["exercise"]==ex}) for ex in index["enabled_form"]},
                                          "models":results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("mlp", "yolo", "both"), required=True)
    parser.add_argument("--audit-approved-by", required=True, help="Explicit human approval of reviewed frame/box audit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    args = parser.parse_args()
    if not args.audit_approved_by.strip():
        raise SystemExit("Approval name cannot be empty")
    index_path = ROOT / "data/derived/training/training_index.json"
    if not index_path.exists():
        raise SystemExit("Run tools.build_training_sets after human review first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    path = path_under(ROOT, args.manifest)
    if sha256_file(path) != index["manifest_sha256"]:
        raise SystemExit("Reviewed manifest changed after export; rebuild training sets")
    rows = read_manifest(path)
    valid_rows(rows, ROOT)
    lookup = {r["sample_id"]:r for r in rows}
    recognition_counts = Counter(lookup[i]["exercise"] for i in index["recognition_train_ids"])
    if set(recognition_counts) != set(EXERCISES) or any(recognition_counts[ex] < 10 for ex in EXERCISES):
        raise SystemExit(f"Need >=10 pose-readable reviewed train frames for all five exercises: {recognition_counts}")
    if not any(r["review_state"] == "accepted" and r["split"] == "val" for r in rows):
        raise SystemExit("Review at least one provenance-disjoint validation group before training")
    if index["enabled_form"] and not index["form_ids_by_split"]["val"]:
        raise SystemExit("Enabled form classes need reviewed validation frames before training both branches")
    if not index["form_training_ids"]:
        print("No form class meets support gate; exercise recognition training only.")
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    save_json(out / "audit_approval.json", {"approved_by":args.audit_approved_by.strip(),
                                            "manifest_sha256":index["manifest_sha256"],
                                            "scope":"reviewed frame and person-box audit for local research training"})
    if args.branch in ("mlp", "both"):
        run_mlp(rows, index, out)
    if args.branch in ("yolo", "both"):
        run_yolo(rows, index, out / "yolo", args.epochs, args.device, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
