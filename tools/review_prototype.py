"""Human frame/box review with OpenCV. Save after every decision.

Keys: v confirm proposed box, b redraw box, e exercise, f form, p phase, c cue-free check,
g validated geometry, a accept, x reject, n next, q/ESC quit.
Acceptance of an incorrect frame prompts for an error description in terminal.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from prototype.schema import EXERCISES, PHASES, box, path_under, read_manifest, valid_rows, write_manifest

ROOT = Path(__file__).resolve().parents[1]
FORMS = ("", "correct", "incorrect")
WINDOW = "Prototype frame review"


def cycle(current, options):
    return options[(options.index(current) + 1) % len(options)] if current in options else options[0]


def screen(image, row, number, total, status_msg=""):
    display = image.copy()
    try:
        bb = box(row)
    except ValueError:
        bb = None
    if bb:
        cv2.rectangle(display, bb[:2], bb[2:], (0, 200, 255), 2)

    scale = min(1.0, 1150 / display.shape[1], 720 / display.shape[0])
    display = cv2.resize(display, None, fx=scale, fy=scale)

    header_h = 180
    canvas_w = max(1080, display.shape[1])
    canvas_h = header_h + display.shape[0]

    pane = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    # Center display horizontally under header
    x_offset = (canvas_w - display.shape[1]) // 2
    pane[header_h:header_h + display.shape[0], x_offset:x_offset + display.shape[1]] = display

    # Divider line
    cv2.line(pane, (0, header_h - 2), (canvas_w, header_h - 2), (60, 60, 60), 1)

    lines = [
        (f"{number}/{total} | ID: {row['sample_id']} | Source: {row['source_name']} | Split: {row['split']}", (220, 220, 220)),
        (f"Hint: {row['exercise_hint']} / {row['form_hint']} (reference only, not truth)", (180, 180, 180)),
        (f"Exercise [e]: {row['exercise'] or '?'}    Form [f]: {row['form'] or 'unrated'}    Phase [p]: {row['phase']}", (0, 220, 255)),
        (f"Box reviewed [v]: {row['bbox_reviewed']}    Cue free [c]: {row['cue_free']}    Geom valid [g]: {row['geometry_validated']}",
         (0, 255, 180) if row['cue_free'] == 'yes' and row['bbox_reviewed'] == 'yes' else (100, 200, 255)),
        ("KEYS: [v] confirm box | [b] redraw | [e] exercise | [f] form | [p] phase | [c] cue | [g] geom", (200, 200, 200)),
        ("      [a] accept | [x] reject | [n] next | [q/ESC] quit", (200, 200, 200)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(pane, text[:125], (15, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    if status_msg:
        cv2.putText(pane, f">> {status_msg}"[:120], (15, 25 + len(lines) * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 1, cv2.LINE_AA)

    return pane, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/prototype_samples.csv")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--exercise", choices=EXERCISES)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--queue", help="Optional CSV with sample_id column, e.g. manifests/prototype_review_priority.csv")
    args = parser.parse_args()
    path = path_under(ROOT, args.manifest)
    rows = read_manifest(path)
    queue = None
    if args.queue:
        with path_under(ROOT, args.queue).open(newline="", encoding="utf-8") as f:
            queue = {r["sample_id"] for r in csv.DictReader(f)}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    index = args.start
    status_msg = ""
    while index < len(rows):
        row = rows[index]
        if queue is not None and row["sample_id"] not in queue:
            index += 1
            continue
        if args.exercise and row["exercise_hint"] != args.exercise:
            index += 1
            continue
        if args.split and row["split"] != args.split:
            index += 1
            continue
        if row["review_state"] != "pending":
            index += 1
            continue
        image = cv2.imread(str(path_under(ROOT, row["image_path"])))
        if image is None:
            raise FileNotFoundError(row["image_path"])
        while True:
            pane, scale = screen(image, row, index+1, len(rows), status_msg)
            cv2.imshow(WINDOW, pane)
            key = cv2.waitKey(0)

            # Check if window was closed via title bar 'X'
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return

            if key == -1:
                continue

            k = key & 0xff
            char = chr(k).lower() if 0 <= k <= 255 and chr(k).isprintable() else ""

            if k == 27 or char == "q":
                cv2.destroyAllWindows()
                return
            if char == "n":
                status_msg = ""
                break
            if char == "e":
                if not row["exercise"] and row["exercise_hint"] in EXERCISES:
                    row["exercise"] = row["exercise_hint"]
                else:
                    row["exercise"] = cycle(row["exercise"], EXERCISES)
                status_msg = f"Exercise set to: {row['exercise']}"
            elif char == "f":
                row["form"] = cycle(row["form"], FORMS)
                row["error_type"] = ""
                status_msg = f"Form set to: {row['form'] or 'unrated'}"
            elif char == "p":
                row["phase"] = cycle(row["phase"], PHASES)
                status_msg = f"Phase set to: {row['phase']}"
            elif char == "c":
                row["cue_free"] = "yes" if row["cue_free"] != "yes" else "no"
                status_msg = f"Cue-free set to: {row['cue_free']}"
            elif char == "g":
                row["geometry_validated"] = "yes" if row["geometry_validated"] != "yes" else "no"
                status_msg = f"Geometry validated set to: {row['geometry_validated']}"
            elif char == "b":
                roi = cv2.selectROI("Select person box, ENTER to confirm", image, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select person box, ENTER to confirm")
                x, y, w, h = [int(v) for v in roi]
                if w > 0 and h > 0:
                    for name, value in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), (x, y, x+w, y+h)):
                        row[name] = str(value)
                    row["bbox_reviewed"] = "yes"
                    status_msg = "Person box updated and confirmed."
                else:
                    status_msg = "Box selection cancelled."
            elif char == "v":
                if box(row) is not None:
                    row["bbox_reviewed"] = "no" if row["bbox_reviewed"] == "yes" else "yes"
                    status_msg = f"Box reviewed set to: {row['bbox_reviewed']}"
                else:
                    status_msg = "No box proposal! Press 'b' to draw person box."
            elif char == "x":
                print(f"\n[REJECT] Sample {row['sample_id']} - enter reason in terminal:")
                reason = input("Reason for rejection: ").strip()
                row.update(review_state="rejected", reviewer=args.reviewer, review_note=reason)
                write_manifest(path, rows)
                print(f"Sample {row['sample_id']} rejected and saved.")
                status_msg = ""
                break
            elif char == "a":
                # Pre-validate before taking inputs
                if not row["exercise"]:
                    row["exercise"] = row["exercise_hint"] if row["exercise_hint"] in EXERCISES else ""
                if not row["exercise"]:
                    status_msg = "Cannot accept: exercise must be selected (press 'e')!"
                    continue
                if box(row) is None:
                    status_msg = "Cannot accept: missing bounding box (press 'b' to draw)!"
                    continue
                if row["bbox_reviewed"] != "yes":
                    status_msg = "Cannot accept: box not confirmed (press 'v' or 'b')!"
                    continue
                if row["cue_free"] != "yes":
                    status_msg = "Cannot accept: cue-free check required (press 'c')!"
                    continue
                if not row["phase"] or row["phase"] == "unknown":
                    status_msg = "Cannot accept: phase must not be unknown (press 'p')!"
                    continue

                if row["form"] == "incorrect":
                    print(f"\n[ACCEPT] Sample {row['sample_id']} has incorrect form.")
                    error_type = input("Observed error type (specific, not folder name): ").strip()
                    if not error_type:
                        status_msg = "Cannot accept incorrect form without error_type!"
                        print(f">> {status_msg}")
                        continue
                    row["error_type"] = error_type

                row.update(review_state="accepted", uncertain="no", reviewer=args.reviewer)
                try:
                    valid_rows([row], ROOT)
                except ValueError as error:
                    row["review_state"] = "pending"
                    row["uncertain"] = "yes"
                    status_msg = f"Cannot accept: {error}"
                    print(f">> {status_msg}")
                    continue

                print("[ACCEPT] Review note (optional in terminal, press ENTER to skip):")
                row["review_note"] = input("Review note (optional): ").strip()
                write_manifest(path, rows)
                print(f"Sample {row['sample_id']} accepted and saved.")
                status_msg = ""
                break
            elif char:
                status_msg = f"Key '{char}' not recognized. See KEYS legend above."
        index += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

