"""Make a SYNTHETIC practice dataset for the Session 6 audit.

NOT real traffic data. It has 100 images and COCO annotations
with deliberate problems for the audit to identify.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

random.seed(7)
rng = np.random.default_rng(7)

root = Path(__file__).resolve().parent
image_dir = root / "data" / "audit" / "images"
ann_dir = root / "data" / "audit" / "annotations"

image_dir.mkdir(parents=True, exist_ok=True)
ann_dir.mkdir(parents=True, exist_ok=True)

W, H = 640, 360

categories = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "motorcycle"},
    {"id": 3, "name": "pedestrian"},
    {"id": 4, "name": "bus"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "autorickshaw"},
]

colour = {
    1: (60, 60, 200),
    2: (60, 200, 60),
    3: (200, 150, 60),
    4: (200, 60, 60),
    5: (120, 120, 120),
    6: (230, 200, 40)
}

DARK = {10, 11, 12}
BLURRY = {20, 21, 22, 23}
CORRUPT = {30, 31}
MISSING = {40, 41}
SIZE_MISMATCH = {50}
NO_OBJECTS = {90, 91, 92, 93, 94}

autorickshaw_images = set(random.sample(range(1, 90), 5))

images = []
annotations = []
ann_id = 1

common_classes = (
    [1] * 45 + [2] * 25 + [3] * 15 +
    [4] * 6 + [5] * 9
)

for i in range(1, 101):

    name = f"img_{i:03d}.jpg"

    images.append({
        "id": i,
        "file_name": name,
        "width": W,
        "height": H
    })

    pixels = np.clip(
        rng.normal(150, 25, (H, W, 3)),
        0, 255
    )

    boxes = []

    n = 0 if i in NO_OBJECTS else random.randint(2, 6)

    classes = [
        random.choice(common_classes)
        for _ in range(n)
    ]

    if i in autorickshaw_images and classes:
        classes[0] = 6

    for cid in classes:

        bw = random.randint(40, 140)
        bh = random.randint(40, 120)

        x = random.randint(0, W - bw)
        y = random.randint(0, H - bh)

        base = np.array(colour[cid], dtype=float)

        pixels[y:y + bh, x:x + bw] = np.clip(
            base + rng.normal(0, 18, (bh, bw, 3)),
            0, 255
        )

        boxes.append((cid, [x, y, bw, bh]))

        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": cid,
            "bbox": [x, y, bw, bh],
            "area": bw * bh,
            "iscrowd": 0
        })

        ann_id += 1

    picture = Image.fromarray(pixels.astype("uint8"))

    if i in DARK:
        picture = Image.fromarray(
            (np.asarray(picture) * 0.2).astype("uint8")
        )

    if i in BLURRY:
        picture = picture.filter(
            ImageFilter.GaussianBlur(radius=6)
        )

    if i in SIZE_MISMATCH:
        picture = picture.resize((W // 2, H // 2))

    if i in MISSING:
        continue

    if i in CORRUPT:
        (image_dir / name).write_bytes(
            b"this is not a jpeg" * 20
        )
        continue

    picture.save(image_dir / name, quality=92)


def add(image_id, category_id, bbox, area=None):

    global ann_id

    annotations.append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": (
            area if area is not None
            else max(bbox[2], 0) * max(bbox[3], 0)
        ),
        "iscrowd": 0
    })

    ann_id += 1


# Deliberate annotation problems

add(60, 1, [100, 100, 0, 50])
add(61, 1, [200, 100, 60, 0])

add(62, 1, [600, 300, 120, 100])
add(63, 2, [-20, 50, 80, 80])
add(64, 3, [300, 330, 50, 60])

add(65, 3, [100, 100, 4, 5])
add(66, 2, [300, 200, 6, 6])

first_67 = next(
    a for a in annotations
    if a["image_id"] == 67
)

add(
    67,
    first_67["category_id"],
    list(first_67["bbox"])
)

add(68, 99, [50, 50, 80, 60])
add(999, 1, [50, 50, 80, 60])

add(
    69, 1, [120, 120, 80, 60],
    area=9000
)

# Save COCO annotation JSON

with open(
    ann_dir / "instances.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump({
        "images": images,
        "annotations": annotations,
        "categories": categories
    }, f)

print(
    f"Created {len(images)} image records and "
    f"{len(annotations)} annotations in data/audit/"
)

print(
    "Reminder: synthetic practice data with deliberate "
    "defects, not real traffic data."
)