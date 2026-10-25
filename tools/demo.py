"""Windows OpenCV demo: 1 MLP, 2 YOLO, q quit. Webcam or video file."""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from prototype.feedback import FeedbackGuard
from prototype.pipelines import MLPPipeline, YOLOPipeline
from prototype.schema import path_under, save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video")
    input_group.add_argument("--camera", type=int)
    parser.add_argument("--pipeline", choices=("mlp", "yolo"), default="mlp")
    parser.add_argument("--pose-task", default="artifacts/pose_landmarker.task")
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    parser.add_argument("--metrics-out", default="artifacts/demo_metrics.json")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="Smoke test only; no displayed-FPS metric")
    args = parser.parse_args()
    pose_task = path_under(ROOT, args.pose_task)
    models = {}
    mlp_path = ROOT / "artifacts/mlp.joblib"
    yolo_exercise = ROOT / "artifacts/yolo/exercise/weights/best.pt"
    yolo_form = ROOT / "artifacts/yolo/form/weights/best.pt"
    if mlp_path.exists():
        models["mlp"] = MLPPipeline(mlp_path, pose_task)
    if yolo_exercise.exists():
        models["yolo"] = YOLOPipeline(yolo_exercise, yolo_form)
    if args.pipeline not in models:
        raise SystemExit(f"{args.pipeline} model absent. Review frames, approve audit, then train before demo.")
    guard = FeedbackGuard(path_under(ROOT, args.manifest), ROOT / "data/derived/training/geometry.json",
                          pose_task if pose_task.exists() else None)
    source = args.camera if args.camera is not None else str((ROOT / args.video).resolve())
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if args.camera is not None else cv2.VideoCapture(source)
    if args.camera is not None and not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open capture: {source}")
    if args.camera is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    window_name = "Fitness posture prototype | 1 MLP | 2 YOLO | q quit"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    active = args.pipeline
    start = time.perf_counter()
    display_times = deque(maxlen=120)
    latencies = []
    counts = {"read":0, "rendered":0, "displayed":0}
    statuses = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            counts["read"] += 1
            t0 = time.perf_counter()
            result = models[active].predict(frame)
            geometry = models[active].last_geometry if active == "mlp" else guard.get_geometry(frame)
            message = guard.present(result, geometry)
            result["feedback"] = message
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            if result["bbox"]:
                cv2.rectangle(frame, tuple(result["bbox"][:2]), tuple(result["bbox"][2:]), (0, 255, 0), 2)
            line = f"{active.upper()} | {result['exercise'] or '-'} | {result['status']} | {result['confidence']:.2f}"
            cv2.putText(frame, line, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
            cv2.putText(frame, message[:95], (12, 57), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            now = time.perf_counter()
            latencies.append((now-t0)*1000)
            counts["rendered"] += 1
            if not args.headless:
                cv2.imshow(window_name, frame)
                counts["displayed"] += 1
                display_times.append(time.perf_counter())
                if len(display_times) >= 2:
                    fps = (len(display_times)-1)/(display_times[-1]-display_times[0])
                    cv2.setWindowTitle(window_name, f"{active.upper()} | displayed {fps:.1f} FPS")
                key = cv2.waitKey(1) & 0xff
                if key == ord("q"):
                    break
                if key == ord("1") and "mlp" in models:
                    active = "mlp"
                if key == ord("2") and "yolo" in models:
                    active = "yolo"
            if args.max_frames and counts["read"] >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        guard.close()
        for model in models.values():
            model.close()
    elapsed = time.perf_counter()-start
    metrics = {"source":str(source), "pipeline_initial":args.pipeline, "headless":args.headless,
               "frames":counts, "elapsed_s":elapsed, "status_counts":statuses,
               "displayed_fps":counts["displayed"]/elapsed if counts["displayed"] else None,
               "processed_fps":counts["rendered"]/elapsed if elapsed else None,
               "latency_ms_median":float(np.median(latencies)) if latencies else None,
               "latency_ms_p95":float(np.percentile(latencies, 95)) if latencies else None}
    save_json(path_under(ROOT, args.metrics_out), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
