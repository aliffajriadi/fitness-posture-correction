import math
import tempfile
import unittest
from pathlib import Path

from prototype.pose import FEATURE_NAMES, angle, pose_geometry, primary_pose, vectorize
from prototype.pose import sha256_file
from prototype.pipelines import YOLOPipeline
from prototype.schema import EXERCISES, FIELDS, FORM_LABELS, assert_no_leakage, assign_splits, box, decode_form_label, encode_form_label, form_support, prediction, valid_rows
from tools.evaluate_prototype import scores


class Landmark:
    def __init__(self, x=0, y=0, z=0, visibility=1):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def row(sid, group="group1", form="correct", split="train"):
    result = dict.fromkeys(FIELDS, "")
    result.update(sample_id=sid, source_path="source.jpg", image_path="image.jpg", source_name="test",
                  parent_id=sid, group_id=group, width="100", height="100", exercise="plank", form=form,
                  phase="static", uncertain="no", bbox_x1="1", bbox_y1="2", bbox_x2="90", bbox_y2="95",
                  review_state="accepted", bbox_reviewed="yes", cue_free="yes", reviewer="human", split=split)
    if form == "incorrect":
        result["error_type"] = "hips_low"
    return result


class PrototypeTests(unittest.TestCase):
    def test_angle_and_visibility(self):
        self.assertAlmostEqual(angle((1, 0), (0, 0), (0, 1)), 90)
        self.assertIsNone(angle((0, 0), (0, 0), (1, 1)))
        landmarks = [Landmark() for _ in range(33)]
        landmarks[11] = Landmark(.3, .2)
        landmarks[13] = Landmark(.5, .2)
        landmarks[15] = Landmark(.5, .4)
        landmarks[23] = Landmark(.3, .5)
        landmarks[24] = Landmark(.5, .5)
        landmarks[27] = Landmark(.3, .8)
        landmarks[28] = Landmark(.5, .8)
        vector, geom = vectorize(landmarks)
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertAlmostEqual(geom["left_elbow"], 90)
        landmarks[13].visibility = .1
        self.assertIsNone(pose_geometry(landmarks)["left_elbow"])

    def test_primary_person_selection(self):
        large = [Landmark(.1 + .8*(i % 2), .1 + .8*((i//2) % 2)) for i in range(33)]
        small = [Landmark(.1 + .1*(i % 2), .1 + .1*((i//2) % 2)) for i in range(33)]
        chosen, status = primary_pose([small, large], 100, 100)
        self.assertIs(chosen, large)
        self.assertEqual(status, "ok")
        _, status = primary_pose([large, large], 100, 100)
        self.assertEqual(status, "multiple_people")

    def test_review_and_box_gate(self):
        sample = row("a")
        self.assertEqual(box(sample), (1, 2, 90, 95))
        self.assertTrue(valid_rows([sample]))
        sample["bbox_reviewed"] = "no"
        with self.assertRaises(ValueError):
            valid_rows([sample])
        sample["bbox_reviewed"] = "yes"
        sample["bbox_x2"] = "101"
        with self.assertRaises(ValueError):
            valid_rows([sample])

    def test_reviewed_image_hash_is_bound_to_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"source.jpg").write_bytes(b"source")
            image = root/"image.jpg"
            image.write_bytes(b"reviewed frame")
            sample = row("a")
            sample["image_sha256"] = sha256_file(image)
            self.assertTrue(valid_rows([sample], root))
            image.write_bytes(b"different frame")
            with self.assertRaises(ValueError):
                valid_rows([sample], root)

    def test_linked_groups_never_leak(self):
        rows = [row("a", "subject_a"), row("b", "subject_a"), row("c", "subject_b")]
        rows[2]["duplicate_component"] = "duplicate"
        rows[0]["duplicate_component"] = "duplicate"
        assign_splits(rows)
        self.assertEqual(len({r["split"] for r in rows}), 1)
        rows[1]["split"] = "test" if rows[0]["split"] != "test" else "train"
        with self.assertRaises(ValueError):
            assert_no_leakage(rows)

    def test_form_thresholds_and_contract(self):
        rows = [row(f"c{i}", f"c{i//5}", "correct") for i in range(10)]
        rows += [row(f"b{i}", f"b{i//5}", "incorrect") for i in range(10)]
        self.assertTrue(form_support(rows)["plank"]["enabled"])
        rows[0]["review_state"] = "pending"
        self.assertFalse(form_support(rows)["plank"]["enabled"])
        result = prediction("plank", "correct", "unsupported_form", .9, (0, 0, 10, 10))
        self.assertIsNone(result["form"])
        self.assertEqual(tuple(result), ("exercise", "form", "status", "confidence", "bbox", "feedback"))

    def test_duplicate_origin_does_not_count_as_two_groups(self):
        rows = [row(f"c{i}", "subject_a" if i < 5 else "subject_b", "correct") for i in range(10)]
        rows += [row(f"b{i}", "subject_c" if i < 5 else "subject_d", "incorrect") for i in range(10)]
        rows[0]["duplicate_component"] = "same-recording"
        rows[5]["duplicate_component"] = "same-recording"
        gate = form_support(rows)["plank"]
        self.assertEqual(gate["conditions"]["correct"]["groups"], 1)
        self.assertFalse(gate["enabled"])

    def test_common_denominator_includes_failure(self):
        rows = [row("a", split="test"), row("b", form="incorrect", split="test")]
        outputs = {"a":prediction(status="no_person"), "b":prediction("plank", "incorrect", "ok", .8)}
        report = scores(rows, outputs)
        self.assertEqual(report["frames"], 2)
        self.assertEqual(report["form_per_class"]["plank"]["confusion"]["correct"]["abstain"], 1)
        with self.assertRaises(ValueError):
            scores(rows, {"a":outputs["a"]})

    def test_yolo_disagreement_abstains(self):
        import numpy as np
        pipeline = YOLOPipeline.__new__(YOLOPipeline)
        pipeline.exercise_model = "exercise"
        pipeline.form_model = "form"
        pipeline.person_model = None
        pipeline.threshold = .6
        pipeline.metadata = {"enabled_form":["push_up"]}
        pipeline._detections = lambda model, image, threshold: (
            [("push_up", .9, [1, 1, 40, 40])] if model == "exercise"
            else [("squat_correct", .8, [1, 1, 40, 40])])
        result = pipeline.predict(np.zeros((50, 50, 3), dtype=np.uint8))
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["form"])

    def test_global_form_label_roundtrip(self):
        self.assertEqual(len(FORM_LABELS), 10)
        for exercise in EXERCISES:
            for form in ("correct", "incorrect"):
                self.assertEqual(decode_form_label(encode_form_label(exercise, form)), (exercise, form))
        with self.assertRaises(ValueError):
            decode_form_label("not_mountain_climber")


if __name__ == "__main__":
    unittest.main()
