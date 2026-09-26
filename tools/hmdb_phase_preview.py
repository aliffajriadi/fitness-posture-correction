"""Five-phase visual screen of candidate original HMDB sit-up clips."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageOps

root = Path(__file__).resolve().parents[1]
files = sorted((root / "data/sources/hmdb51/hmdb51_org/situp").rglob("*.avi"))
out = root / "manifests/previews"
for index in [3, 30, 44, 50, 62, 97]:
    path = files[index]
    canvas = Image.new("RGB", (5*256, 230), "white")
    draw = ImageDraw.Draw(canvas)
    for j, frac in enumerate([0, .25, .5, .75, .95]):
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((count-1)*frac))
        ok, frame = cap.read()
        cap.release()
        if ok:
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            canvas.paste(ImageOps.contain(im, (256, 192)), (j*256, 0))
        draw.text((j*256+2, 195), f"{index:03} {frac:.2f}", fill="black")
    canvas.save(out / f"HMDB_situp_phase_{index:03}.jpg", quality=90)
    print(index, path.name, flush=True)
