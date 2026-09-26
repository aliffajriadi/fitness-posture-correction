"""The one pose feature specification used during training and live inference."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from .schema import EXERCISES

VISIBILITY = .50
ANGLE_TRIPLES = {
    "left_elbow": (11, 13, 15), "right_elbow": (12, 14, 16),
    "left_hip": (11, 23, 25), "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27), "right_knee": (24, 26, 28),
    "left_body": (11, 23, 27), "right_body": (12, 24, 28),
}
SPEC_VERSION = "pose-v1-33xyz-visibility-8angles-2torso"
FEATURE_NAMES = tuple(f"lm{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "visibility")) + \
                tuple(f"angle_{k}" for k in ANGLE_TRIPLES) + \
                ("torso_horizontal", "torso_vertical")


def angle(a, b, c):
    va = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    vc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vc)
    if denom < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(va, vc) / denom, -1, 1))))


def pose_geometry(landmarks):
    """Return angle values only when all contributing points are visible."""
    geom = {}
    for name, (a, b, c) in ANGLE_TRIPLES.items():
        if all(landmarks[i].visibility >= VISIBILITY for i in (a, b, c)):
            geom[name] = angle((landmarks[a].x, landmarks[a].y), (landmarks[b].x, landmarks[b].y),
                               (landmarks[c].x, landmarks[c].y))
        else:
            geom[name] = None
    for side, shoulder, hip in (("left", 11, 23), ("right", 12, 24)):
        if landmarks[shoulder].visibility >= VISIBILITY and landmarks[hip].visibility >= VISIBILITY:
            dx, dy = landmarks[hip].x - landmarks[shoulder].x, landmarks[hip].y - landmarks[shoulder].y
            geom[f"{side}_torso_horizontal"] = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
            geom[f"{side}_torso_vertical"] = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        else:
            geom[f"{side}_torso_horizontal"] = None
            geom[f"{side}_torso_vertical"] = None
        ankle = 27 if side == "left" else 28
        if all(landmarks[i].visibility >= VISIBILITY for i in (shoulder, hip, ankle)):
            sx, sy = landmarks[shoulder].x, landmarks[shoulder].y
            ax, ay = landmarks[ankle].x, landmarks[ankle].y
            hx, hy = landmarks[hip].x, landmarks[hip].y
            along = ((hx-sx)*(ax-sx)+(hy-sy)*(ay-sy))/max((ax-sx)**2+(ay-sy)**2, 1e-9)
            expected_y = sy + along*(ay-sy)
            geom[f"{side}_hip_line_offset"] = (hy-expected_y)/max(math.hypot(ax-sx, ay-sy), 1e-9)
        else:
            geom[f"{side}_hip_line_offset"] = None
    geom["torso_horizontal"] = visible_mean(geom, ("left_torso_horizontal", "right_torso_horizontal"))
    geom["torso_vertical"] = visible_mean(geom, ("left_torso_vertical", "right_torso_vertical"))
    geom["hip_line_offset"] = visible_mean(geom, ("left_hip_line_offset", "right_hip_line_offset"))
    return geom


def visible_mean(geom, keys):
    values = [geom[k] for k in keys if geom.get(k) is not None]
    return float(np.mean(values)) if values else None


def vectorize(landmarks):
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    scale = max(math.hypot(hip_x-shoulder_x, hip_y-shoulder_y), .08)
    vector = []
    for lm in landmarks:
        if lm.visibility >= VISIBILITY:
            vector += [(lm.x-hip_x)/scale, (lm.y-hip_y)/scale, lm.z/scale, lm.visibility]
        else:
            vector += [0.0, 0.0, 0.0, lm.visibility]
    geom = pose_geometry(landmarks)
    vector += [(geom[k] / 180 if geom[k] is not None else -1.0) for k in ANGLE_TRIPLES]
    vector += [(geom[k] / 90 if geom[k] is not None else -1.0) for k in ("torso_horizontal", "torso_vertical")]
    return np.asarray(vector, dtype=np.float32), geom


def phase_for(exercise, geom):
    if exercise == "plank":
        return "static" if visible_mean(geom, ("left_body", "right_body")) is not None else "unknown"
    joints = {"push_up": ("left_elbow", "right_elbow"),
              "sit_up": ("left_hip", "right_hip"),
              "squat": ("left_knee", "right_knee"),
              "mountain_climber": ("left_knee", "right_knee")}
    values = [geom[k] for k in joints[exercise] if geom.get(k) is not None]
    if not values:
        return "unknown"
    value = min(values) if exercise == "mountain_climber" else float(np.mean(values))
    if exercise == "mountain_climber":
        return "bottom" if value < 110 else "top" if value > 150 else "middle"
    if exercise == "sit_up":
        return "bottom" if value < 110 else "top" if value > 155 else "middle"
    return "bottom" if value < 125 else "top" if value > 155 else "middle"


def bbox_from_landmarks(landmarks, width, height):
    points = [(lm.x * width, lm.y * height) for lm in landmarks if lm.visibility >= VISIBILITY]
    if not points:
        return None
    xs, ys = zip(*points)
    pad = .10 * max(max(xs)-min(xs), max(ys)-min(ys))
    return [max(0, int(min(xs)-pad)), max(0, int(min(ys)-pad)),
            min(width, int(max(xs)+pad)), min(height, int(max(ys)+pad))]


def primary_pose(poses, width, height):
    """Ignore tiny background poses; abstain for two comparably prominent people."""
    candidates = []
    for pose in poses:
        bb = bbox_from_landmarks(pose, width, height)
        if bb:
            area = max(0, bb[2]-bb[0])*max(0, bb[3]-bb[1])
            candidates.append((area, pose))
    if not candidates:
        return None, "no_person"
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] >= .35*candidates[0][0]:
        return None, "multiple_people"
    return candidates[0][1], "ok"


class PoseExtractor:
    def __init__(self, task_path):
        import mediapipe as mp
        self.mp = mp
        self.task_path = Path(task_path)
        if not self.task_path.is_file():
            raise FileNotFoundError(f"Download a MediaPipe Pose Landmarker .task model: {self.task_path}")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.task_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=2)
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self.landmarker.detect(image).pose_landmarks

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
