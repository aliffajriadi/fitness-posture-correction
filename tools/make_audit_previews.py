"""Make read-only visual inspection sheets from source media."""
from pathlib import Path
import csv
import math
import re

import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests" / "previews"
OUT.mkdir(parents=True, exist_ok=True)


def contact(name, pairs, cols=6, width=180, height=135):
    if not pairs:
        return
    canvas = Image.new("RGB", (cols * width, math.ceil(len(pairs)/cols) * (height+42)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, (im, label) in enumerate(pairs):
        x, y = j % cols * width, j // cols * (height+42)
        canvas.paste(ImageOps.contain(im, (width, height)), (x, y))
        draw.text((x+2, y+height+2), label[:27], fill="black")
        draw.text((x+2, y+height+19), label[27:54], fill="black")
    canvas.save(OUT / name, quality=85)


def frame(path, frac):
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((count-1)*frac))
    ok, bgr = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if ok else None


for source in ["Mountain_Climbers_Pose_Detection", "Plank_detection"]:
    files = sorted((ROOT / "data" / "sources" / source).rglob("*.jpg"))
    for page in range(math.ceil(len(files)/48)):
        pairs = []
        for path in files[page*48:(page+1)*48]:
            label_path = path.parent.parent / "labels" / (path.stem + ".txt")
            label = label_path.read_text().splitlines()[0].split()[0] if label_path.exists() and label_path.read_text().strip() else "?"
            with Image.open(path) as im:
                pairs.append((im.convert("RGB"), f"{label} {path.stem[:24]}"))
        contact(source + f"_{page+1:02}.jpg", pairs)
    print(source, len(files), "images", flush=True)

for exercise in ["situp", "pushup"]:
    hmdb = sorted((ROOT / "data" / "sources" / "hmdb51" / "hmdb51_org" / exercise).rglob("*.avi"))
    for page in range(math.ceil(len(hmdb)/48)):
        pairs = []
        for j, path in enumerate(hmdb[page*48:(page+1)*48], page*48):
            im = frame(path, 0.5)
            if im:
                pairs.append((im, f"{j:03} {path.stem[:42]}"))
        contact(f"HMDB_{exercise}_{page+1:02}.jpg", pairs)
    print("HMDB", exercise, len(hmdb), "clips", flush=True)

base = ROOT / "data" / "sources" / "Dataset Subject-wise"
for ex in ["push_up", "squat"]:
    paths = sorted(base.glob(f"subject_*/{ex}/*/side/*.mp4"))
    for label in ["good", "bad"]:
        chosen = [p for p in paths if p.parts[-3] == label and p.parts[-5] in {"subject_001", "subject_005", "subject_010", "subject_015", "subject_020", "subject_024", "subject_026"}]
        for p in chosen:
            pairs = []
            for f in [0, .25, .5, .75, .95]:
                im = frame(p, f)
                if im:
                    pairs.append((im, f"{p.parts[-5]} {label} {f:.2f}"))
            contact(f"Mendeley_{ex}_{label}_{p.parts[-5]}.jpg", pairs, cols=5, width=256, height=192)
    print("Mendeley", ex, len(paths), "side clips", flush=True)

zen = ROOT / "data" / "sources" / "Squat_Dataset_Zenodo"
for label in ["Good", "Bad back", "Bad heel"]:
    files = sorted((zen / "train" / label).glob("*.jpg"))
    chosen = [files[i] for i in range(0, len(files), max(1, len(files)//36))][:36]
    pairs = []
    for p in chosen:
        with Image.open(p) as im:
            pairs.append((im.convert("RGB"), p.stem[:35]))
    contact("Zenodo_"+label.replace(" ","_")+".jpg", pairs)
    print("Zenodo", label, len(files), flush=True)
