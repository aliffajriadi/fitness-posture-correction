"""Read-only probe of tight person crops in the local mountain-climber export."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/Mountain_Climbers_Pose_Detection"
PREVIEW = ROOT / "manifests/previews/mountain_climber_bbox_crop_probe.jpg"
SUMMARY = ROOT / "manifests/mountain_climber_crop_probe.json"
LABELS = {0: "correct", 1: "incorrect"}


def inspect(image_path: Path) -> dict:
    label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"Expected one target box: {label_path}")
    label, cx, cy, bw, bh = map(float, rows[0])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    left = max(0, round((cx - bw / 2) * width))
    top = max(0, round((cy - bh / 2) * height))
    right = min(width, round((cx + bw / 2) * width))
    bottom = min(height, round((cy + bh / 2) * height))
    if left >= right or top >= bottom:
        raise ValueError(f"Empty crop: {image_path}")
    crop = image.crop((left, top, right, bottom))
    hsv = np.asarray(crop.convert("HSV"))
    colorful_fraction = float(np.mean((hsv[:, :, 1] >= 32) & (hsv[:, :, 2] >= 25)))
    rgb = np.asarray(crop).astype(np.int16)
    red_fraction = float(
        np.mean(
            (rgb[:, :, 0] >= 120)
            & (rgb[:, :, 0] >= 2 * rgb[:, :, 1])
            & (rgb[:, :, 0] >= 2 * rgb[:, :, 2])
        )
    )
    return {
        "path": image_path,
        "family": image_path.name.split("_", 1)[0],
        "class": LABELS[int(label)],
        "image": image,
        "crop": crop,
        "crop_width": right - left,
        "crop_height": bottom - top,
        "box_area_fraction": (right - left) * (bottom - top) / (width * height),
        "colorful_fraction": colorful_fraction,
        "red_fraction": red_fraction,
    }


def main() -> None:
    items = [
        inspect(path)
        for split in ("train", "valid")
        for path in sorted((SOURCE / split / "images").glob("*.jpg"))
    ]
    groups = defaultdict(list)
    for item in items:
        groups[(item["family"], item["class"])].append(item)

    summary = {"total_images": len(items), "groups": {}}
    selected = []
    for key in sorted(groups):
        group = groups[key]
        label = "/".join(key)
        summary["groups"][label] = {
            "images": len(group),
            "median_crop_width_px": median(x["crop_width"] for x in group),
            "median_crop_height_px": median(x["crop_height"] for x in group),
            "median_box_area_fraction": round(median(x["box_area_fraction"] for x in group), 4),
            "median_colorful_fraction_inside_box": round(
                median(x["colorful_fraction"] for x in group), 4
            ),
            "crops_with_red_overlay_candidate": sum(x["red_fraction"] > 0.002 for x in group),
            "red_overlay_candidate_examples": [
                x["path"].name for x in group if x["red_fraction"] > 0.002
            ][:8],
        }
        selected.extend(group[round(i * (len(group) - 1) / 3)] for i in range(4))

    tile_width, tile_height = 235, 266
    sheet = Image.new("RGB", (tile_width * 4, tile_height * ((len(selected) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, item in enumerate(selected):
        x = (idx % 4) * tile_width
        y = (idx // 4) * tile_height
        for col, image in enumerate((item["image"], item["crop"])):
            thumb = ImageOps.contain(image, (110, 215))
            sheet.paste(thumb, (x + 3 + col * 115, y + 3 + (215 - thumb.height) // 2))
        draw.text((x + 3, y + 222), f"{item['family']} {item['class']}", fill="black")
        draw.text((x + 3, y + 239), item["path"].name[:31], fill="black")

    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(PREVIEW, quality=88)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(PREVIEW)


if __name__ == "__main__":
    main()
