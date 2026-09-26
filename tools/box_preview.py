"""Visual spot check of source YOLO boxes, without editing source images."""
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageOps

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / "manifests/source_inventory.csv").open(newline="", encoding="utf-8")))
chosen = []
for source, kind, limit in [("Plank_detection", "single", 8), ("Plank_detection", "multi", 8), ("Mountain_Climbers_Pose_Detection", "single", 8)]:
    subset = [r for r in rows if r["source_name"] == source and r["asset_type"] == "image" and ((";" not in r["source_label"]) if kind == "single" else (";" in r["source_label"]))]
    step = max(1, len(subset)//limit)
    chosen.extend(subset[::step][:limit])
canvas = Image.new("RGB", (4*256, 6*228), "white")
draw = ImageDraw.Draw(canvas)
for i, row in enumerate(chosen):
    path = root / row["relative_path"]
    label_file = path.parent.parent / "labels" / (path.stem+".txt")
    with Image.open(path) as src:
        im = src.convert("RGB")
    box_draw = ImageDraw.Draw(im)
    for line in label_file.read_text().splitlines():
        cls, cx, cy, w, h = [float(v) for v in line.split()]
        x1, y1, x2, y2 = (cx-w/2)*im.width, (cy-h/2)*im.height, (cx+w/2)*im.width, (cy+h/2)*im.height
        box_draw.rectangle((x1,y1,x2,y2), outline="red", width=5)
        box_draw.text((x1,y1), str(int(cls)), fill="red")
    x, y = (i%4)*256, (i//4)*228
    canvas.paste(ImageOps.contain(im,(256,192)),(x,y))
    draw.text((x+2,y+195), row["source_name"][:7]+" "+row["source_label"][:27],fill="black")
canvas.save(root / "manifests/previews/yolo_boxes_spotcheck.jpg", quality=90)
print(len(chosen), "box overlays")
