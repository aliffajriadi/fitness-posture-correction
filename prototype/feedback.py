"""Presentation-only geometry guard and cautiously calibrated reason messages."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .pose import PoseExtractor, phase_for, pose_geometry, primary_pose, visible_mean
from .schema import component_ids, read_manifest

INDICATORS = {
    "plank": (("left_body", "right_body"), ("hip_line_offset",)),
    "mountain_climber": (("left_body", "right_body"), ("left_knee", "right_knee")),
    "push_up": (("left_elbow", "right_elbow"), ("left_body", "right_body")),
    "sit_up": (("left_hip", "right_hip"),),
    "squat": (("left_knee", "right_knee"), ("left_hip", "right_hip"), ("torso_vertical",)),
}
REASONS = {
    "hips_high": ("hip_line_offset", -1, "Pinggul terlalu tinggi terhadap garis tubuh"),
    "hips_low": ("hip_line_offset", 1, "Pinggul turun terhadap garis tubuh"),
    "torso_forward": ("torso_vertical", 1, "Batang tubuh condong ke depan"),
}


class FeedbackGuard:
    def __init__(self, manifest, geometry_path, pose_task=None):
        self.rows = read_manifest(manifest)
        self.component_ids = component_ids(self.rows)
        self.geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8")) if Path(geometry_path).exists() else {}
        self.pose = PoseExtractor(pose_task) if pose_task else None
        self.validated = defaultdict(set)
        for row in self.rows:
            if row["review_state"] == "accepted" and row["geometry_validated"] == "yes":
                self.validated[(row["exercise"], row["phase"])].add(row["form"])
        self.thresholds = self._calibrate()

    def _calibrate(self):
        thresholds = {}
        for exercise, phase in self.validated:
            candidates = [r for r in self.rows if r["review_state"] == "accepted"
                          and r["geometry_validated"] == "yes" and r["exercise"] == exercise and r["phase"] == phase]
            for error_type, (feature, direction, message) in REASONS.items():
                good_rows = [r for r in candidates if r["form"] == "correct"]
                good = [self.geometry.get(r["sample_id"], {}).get(feature) for r in good_rows]
                bad_rows = [r for r in candidates if r["form"] == "incorrect" and r["error_type"] == error_type]
                bad = [self.geometry.get(r["sample_id"], {}).get(feature) for r in bad_rows]
                good = [v for v in good if v is not None]
                bad = [v for v in bad if v is not None]
                if len(good) < 5 or len(bad) < 5 or len({self.component_ids[r["sample_id"]] for r in bad_rows}) < 2 \
                   or len({self.component_ids[r["sample_id"]] for r in good_rows}) < 2:
                    continue
                a, b = float(np.median(good)), float(np.median(bad))
                gap = (b-a)*direction
                if gap <= (.04 if feature == "hip_line_offset" else 8):
                    continue
                thresholds[(exercise, phase, error_type)] = (feature, direction, (a+b)/2, message)
        return thresholds

    def get_geometry(self, bgr):
        if self.pose is None:
            return None
        poses = self.pose.detect(bgr)
        chosen, status = primary_pose(poses, bgr.shape[1], bgr.shape[0])
        return pose_geometry(chosen) if status == "ok" else None

    def present(self, result, geometry):
        if result["status"] != "ok":
            return result["feedback"] or result["status"]
        exercise = result["exercise"]
        if geometry is None:
            return "Form belum ditampilkan: indikator geometri tidak tersedia"
        phase = phase_for(exercise, geometry)
        if phase == "unknown" or self.validated.get((exercise, phase), set()) != {"correct", "incorrect"}:
            return "Form belum ditampilkan: fase/geometri belum tervalidasi"
        if any(visible_mean(geometry, indicators) is None for indicators in INDICATORS[exercise]):
            return "Form belum ditampilkan: landmark penting tidak terlihat"
        if result["form"] == "correct":
            return "Form benar"
        for (ex, ph, _), (feature, direction, cutoff, message) in self.thresholds.items():
            if ex == exercise and ph == phase and geometry.get(feature) is not None:
                if (geometry[feature]-cutoff)*direction > 0:
                    return "Form salah: " + message
        return "Form salah; alasan spesifik belum tervalidasi"

    def close(self):
        if self.pose is not None:
            self.pose.close()
