"""VisionPort - Dataset Audit (Session 6)"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Thresholds
MIN_BOX_SIDE = 10
DARK_MEAN = 50
BLUR_VAR = 100
RARE_SHARE = 0.05
AREA_TOLERANCE = 0.05


# Options
parser = argparse.ArgumentParser(description="Audit a COCO-format dataset")
parser.add_argument("--ann", default="data/audit/annotations/instances.json")
parser.add_argument("--images", default="data/audit/images")
parser.add_argument("--out", default="docs")
args = parser.parse_args()

ann_path = Path(args.ann)
img_dir = Path(args.images)
out_dir = Path(args.out)

if not ann_path.exists():
    raise SystemExit(f"Annotation file not found: {ann_path}")

out_dir.mkdir(parents=True, exist_ok=True)
report = []


def say(line=""):
    print(line)
    report.append(line)


# Load dataset
with open(ann_path, encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

cat_name = {c["id"]: c["name"] for c in categories}
img_by_id = {im["id"]: im for im in images}


say("# VisionPort dataset audit")
say()
say(f"Annotation file : {ann_path}")
say(f"Image folder    : {img_dir}")


# 1. Overview
say("\n## 1. Overview")
say(f"Images: {len(images)} | Annotations: {len(annotations)} | Categories: {len(categories)}")

by_image = defaultdict(list)

for a in annotations:
    by_image[a["image_id"]].append(a)

no_ann = [im["id"] for im in images if not by_image.get(im["id"])]
say(f"Images with no annotation: {len(no_ann)} {no_ann[:10]}")

sizes = Counter((im["width"], im["height"]) for im in images)
say("Image sizes in the JSON: " +
    ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common()))


# 2. Class distribution
say("\n## 2. Class distribution")

counts = Counter(
    a["category_id"]
    for a in annotations
    if a["category_id"] in cat_name
)

total = sum(counts.values())

say(f"{'id':>3} {'class':<14}{'count':>6} {'share':>6}")

for cid, name in sorted(cat_name.items()):
    n = counts.get(cid, 0)
    share = n / total if total else 0
    say(f"{cid:>3} {name:<14}{n:>6} {share:>6.1%} {'#' * round(share * 50)}")

if counts:
    biggest = max(counts.values())
    smallest = min(counts.values())
    say(f"Imbalance ratio (largest / smallest class): {biggest / smallest:.1f} : 1")

rare = [
    cat_name[c]
    for c in cat_name
    if counts.get(c, 0) / max(total, 1) < RARE_SHARE
]

say(f"Rare classes (under {RARE_SHARE:.0%} of annotations): {rare if rare else 'none'}")


# 3. Annotation validation
say("\n## 3. Annotation validation")

issues = defaultdict(list)

seen_ids = Counter(a["id"] for a in annotations)

seen_boxes = Counter(
    (a["image_id"], a["category_id"], tuple(a["bbox"]))
    for a in annotations
)

for a in annotations:
    aid = a["id"]
    box = a.get("bbox", [])

    if seen_ids[aid] > 1:
        issues["duplicate annotation id"].append(aid)

    if a["image_id"] not in img_by_id:
        issues["points to a missing image"].append(aid)

    if a["category_id"] not in cat_name:
        issues["points to an unknown category"].append(aid)

    if len(box) != 4:
        issues["bbox does not have 4 numbers"].append(aid)
        continue

    x, y, w, h = box

    if w <= 0 or h <= 0:
        issues["zero or negative width/height"].append(aid)
    elif min(w, h) < MIN_BOX_SIDE:
        issues[f"tiny box (side under {MIN_BOX_SIDE} px)"].append(aid)

    if a["image_id"] in img_by_id:
        im = img_by_id[a["image_id"]]

        if x < 0 or y < 0 or x + w > im["width"] or y + h > im["height"]:
            issues["box outside the image"].append(aid)

    if w > 0 and h > 0:
        if abs(a.get("area", w * h) - w * h) > AREA_TOLERANCE * w * h:
            issues["area differs from width x height"].append(aid)

    if seen_boxes[(a["image_id"], a["category_id"], tuple(box))] > 1:
        issues["duplicate box"].append(aid)

if not issues:
    say("No annotation problems found.")

for problem, ids in sorted(issues.items()):
    say(f"- {problem}: {len(ids)} e.g. annotation ids {ids[:5]}")


# 4. Image files and quality
say("\n## 4. Image files and quality")

missing = []
corrupt = []
wrong_size = []
dark = []
blurry = []
quality = []

for im in images:
    path = img_dir / im["file_name"]

    if not path.exists():
        missing.append(im["id"])
        continue

    try:
        with Image.open(path) as picture:
            picture = picture.convert("RGB")
    except Exception:
        corrupt.append(im["id"])
        continue

    if picture.size != (im["width"], im["height"]):
        wrong_size.append(im["id"])

    grey = np.asarray(picture.convert("L"), dtype=float)
    brightness = grey.mean()

    laplacian = (
        4 * grey[1:-1, 1:-1]
        - grey[:-2, 1:-1]
        - grey[2:, 1:-1]
        - grey[1:-1, :-2]
        - grey[1:-1, 2:]
    )

    sharpness = laplacian.var()
    quality.append((im["id"], brightness, sharpness))

    if brightness < DARK_MEAN:
        dark.append(im["id"])

    if sharpness < BLUR_VAR:
        blurry.append(im["id"])


for label, ids in [
    ("file missing", missing),
    ("file cannot be opened (corrupt)", corrupt),
    ("size differs from the JSON", wrong_size),
    (f"too dark (mean brightness under {DARK_MEAN})", dark),
    (f"blurry (sharpness under {BLUR_VAR})", blurry)
]:
    say(f"- {label}: {len(ids)} e.g. image ids {ids[:8]}")


# 5. Object sizes
say("\n## 5. Object size mix (COCO definition by pixel area)")

small = 0
medium = 0
large = 0

for a in annotations:
    box = a.get("bbox", [])

    if len(box) == 4 and box[2] > 0 and box[3] > 0:
        area = box[2] * box[3]

        if area < 32 * 32:
            small += 1
        elif area < 96 * 96:
            medium += 1
        else:
            large += 1

say(f"small (under 32x32): {small} | medium (under 96x96): {medium} | large: {large}")


# 6. Charts
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

names = [cat_name[c] for c in sorted(cat_name)]

ax[0].bar(names, [counts.get(c, 0) for c in sorted(cat_name)])
ax[0].set_title("Annotations per class")
ax[0].tick_params(axis="x", rotation=30)

ax[1].bar(
    ["small", "medium", "large"],
    [small, medium, large]
)
ax[1].set_title("Object size mix")

if quality:
    ids, bright, sharp = zip(*quality)
    flagged = [i in dark or i in blurry for i in ids]

    ax[2].scatter(
        bright,
        np.maximum(sharp, 0.1),
        c=["red" if f else "blue" for f in flagged],
        s=18
    )

    ax[2].axvline(DARK_MEAN, color="grey", linestyle="--")
    ax[2].axhline(BLUR_VAR, color="grey", linestyle="--")
    ax[2].set_yscale("log")

ax[2].set_xlabel("brightness")
ax[2].set_ylabel("sharpness (log)")
ax[2].set_title("Image quality (red = flagged)")

plt.tight_layout()
plt.savefig(out_dir / "audit_charts.png", dpi=110)


# 7. Summary
say("\n## 6. Summary")

n_ann_problems = len({
    i for ids in issues.values() for i in ids
})

n_img_problems = len(
    set(missing) |
    set(corrupt) |
    set(wrong_size) |
    set(dark) |
    set(blurry)
)

say(f"Annotations with at least one problem: {n_ann_problems} of {len(annotations)}")
say(f"Images with at least one problem     : {n_img_problems} of {len(images)}")

say(
    "Suggested actions: remove or repair the flagged annotations, replace missing or "
    "corrupt images, review dark and blurry images by eye, and collect more examples "
    "of rare classes before training."
)

(out_dir / "audit_report.md").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print(f"\nSaved: {out_dir / 'audit_report.md'} and {out_dir / 'audit_charts.png'}")