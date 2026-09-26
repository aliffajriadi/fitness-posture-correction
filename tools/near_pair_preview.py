"""Contact sheet for high-priority pHash candidates; read-only raw data."""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / "manifests/near_duplicate_candidates.csv").open(newline="", encoding="utf-8")))
chosen = []
for source, mode in [("Squat_Dataset_Zenodo", "class"), ("Mountain_Climbers_Pose_Detection", "split")]:
    subset = [r for r in rows if r["source_a"] == source and (r["label_a"] != r["label_b"] if mode == "class" else r["split_a"] != r["split_b"])]
    seen = set()
    for r in subset:
        if r["path_a"] in seen:
            continue
        seen.add(r["path_a"])
        chosen.append(r)
        if len(seen) == 8:
            break
canvas = Image.new("RGB", (4*256, 8*226), "white")
draw = ImageDraw.Draw(canvas)
for i, r in enumerate(chosen):
    for side, key in enumerate(["a", "b"]):
        with Image.open(root / r["path_"+key]) as src:
            im = ImageOps.contain(src.convert("RGB"), (256, 184))
        x, y = (i%2*2+side)*256, (i//2)*226
        canvas.paste(im,(x,y))
        draw.text((x+2,y+185), r["label_"+key]+" "+r["split_"+key], fill="black")
        draw.text((x+2,y+202), Path(r["path_"+key]).name[:31], fill="black")
canvas.save(root / "manifests/previews/phash_pair_spotcheck.jpg", quality=90)
print(len(chosen), "candidate pairs")
