"""Two independent two-stage inference pipelines with one result schema."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .pose import PoseExtractor, bbox_from_landmarks, phase_for, primary_pose, vectorize
from .schema import EXERCISES, FORMS, decode_form_label, prediction


def _confidence(model, vector, label):
    probabilities = model.predict_proba(vector.reshape(1, -1))[0]
    scores = dict(zip(model.classes_ if hasattr(model, "classes_") else model[-1].classes_, probabilities))
    return float(scores.get(label, 0))


class MLPPipeline:
    def __init__(self, model_path, pose_task, threshold=.60):
        import joblib
        self.models = joblib.load(model_path)
        self.metadata = json.loads(Path(model_path).with_name("mlp_metadata.json").read_text(encoding="utf-8"))
        self.pose = PoseExtractor(pose_task)
        self.threshold = threshold
        self.last_geometry = None

    def predict(self, bgr):
        self.last_geometry = None
        poses = self.pose.detect(bgr)
        landmarks, status = primary_pose(poses, bgr.shape[1], bgr.shape[0])
        if status == "no_person":
            return prediction(status="no_person", feedback="Pose tidak terdeteksi")
        if status == "multiple_people":
            return prediction(status="multiple_people", feedback="Lebih dari satu orang")
        vector, geom = vectorize(landmarks)
        self.last_geometry = geom
        exercise_model = self.models["exercise"]
        exercise = str(exercise_model.predict(vector.reshape(1, -1))[0])
        confidence = _confidence(exercise_model, vector, exercise)
        bbox = bbox_from_landmarks(landmarks, bgr.shape[1], bgr.shape[0])
        if confidence < self.threshold:
            return prediction(exercise=exercise, status="uncertain", confidence=confidence, bbox=bbox,
                              feedback="Gerakan belum pasti")
        form_model = self.models["form"].get(exercise)
        if form_model is None:
            return prediction(exercise=exercise, status="unsupported_form", confidence=confidence, bbox=bbox,
                              feedback="Form belum dinilai")
        phase = phase_for(exercise, geom)
        if phase not in self.metadata["reviewed_phases"].get(exercise, []):
            return prediction(exercise=exercise, status="waiting_phase", confidence=confidence, bbox=bbox,
                              feedback="Menunggu fase gerakan yang telah direview")
        form = str(form_model.predict(vector.reshape(1, -1))[0])
        form_conf = _confidence(form_model, vector, form)
        if form_conf < self.threshold:
            return prediction(exercise=exercise, status="uncertain", confidence=min(confidence, form_conf), bbox=bbox,
                              feedback="Penilaian form belum pasti")
        return prediction(exercise=exercise, form=form, status="ok", confidence=min(confidence, form_conf), bbox=bbox)

    def close(self):
        self.pose.close()


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb-intersection else 0


class YOLOPipeline:
    def __init__(self, exercise_weights, form_weights=None, threshold=.60):
        config = Path(exercise_weights).parents[3] / "ultralytics_config"
        config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
        from ultralytics import YOLO
        self.exercise_model = YOLO(str(exercise_weights))
        self.form_model = YOLO(str(form_weights)) if form_weights and Path(form_weights).exists() else None
        self.threshold = threshold
        metadata = Path(exercise_weights).parents[2] / "yolo_metadata.json"
        self.metadata = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
        pretrained = Path(exercise_weights).parents[2] / "yolo12n_pretrained.pt"
        self.person_model = YOLO(str(pretrained)) if pretrained.exists() else None

    @staticmethod
    def _detections(model, bgr, threshold):
        result = model.predict(bgr, conf=threshold, agnostic_nms=True, max_det=10, verbose=False)[0]
        rows = []
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            rows.append((label, float(box.conf.item()), [int(round(v)) for v in box.xyxy[0].tolist()]))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def predict(self, bgr):
        people = None
        if self.person_model is not None:
            people = self.person_model.predict(bgr, classes=[0], conf=.30, verbose=False)[0].boxes
            areas = sorted((float((b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1])) for b in people), reverse=True)
            if len(areas) > 1 and areas[1] >= .35*areas[0]:
                return prediction(status="multiple_people", feedback="Lebih dari satu orang yang sama-sama tampak jelas")
        detections = self._detections(self.exercise_model, bgr, self.threshold)
        if not detections:
            if people is not None and len(people):
                return prediction(status="uncertain", feedback="Orang terlihat, gerakan belum dikenali")
            return prediction(status="no_person", feedback="Orang tidak terdeteksi")
        # Overlapping duplicate class predictions count as one person; separated boxes count as multiple.
        distinct = [detections[0]]
        for detected in detections[1:]:
            if all(_iou(detected[2], other[2]) < .70 for other in distinct):
                distinct.append(detected)
        if len(distinct) > 1:
            return prediction(status="multiple_people", confidence=distinct[0][1], feedback="Lebih dari satu orang")
        exercise, confidence, bbox = distinct[0]
        if exercise not in EXERCISES:
            return prediction(status="uncertain", confidence=confidence, bbox=bbox, feedback="Label gerakan tak dikenal")
        if exercise not in self.metadata.get("enabled_form", []) or self.form_model is None:
            return prediction(exercise=exercise, status="unsupported_form", confidence=confidence, bbox=bbox,
                              feedback="Form belum dinilai")
        forms = [d for d in self._detections(self.form_model, bgr, self.threshold) if _iou(d[2], bbox) >= .50]
        if not forms:
            return prediction(exercise=exercise, status="uncertain", confidence=confidence, bbox=bbox,
                              feedback="Penilai form tidak menemukan orang yang sama")
        label, form_conf, _ = forms[0]
        try:
            candidate_exercise, candidate_form = decode_form_label(label)
        except ValueError:
            return prediction(exercise=exercise, status="uncertain", confidence=min(confidence, form_conf), bbox=bbox)
        if candidate_exercise != exercise or candidate_form not in FORMS:
            return prediction(exercise=exercise, status="uncertain", confidence=min(confidence, form_conf), bbox=bbox,
                              feedback="Pengenal gerakan dan penilai form tidak sepakat")
        # A detector alone cannot validate phase geometry; feedback stays empty.
        return prediction(exercise=exercise, form=candidate_form, status="ok",
                          confidence=min(confidence, form_conf), bbox=bbox)

    def close(self):
        pass
