from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preview YOLO segmentation labels.")
    parser.add_argument("--images", type=Path, default=Path("data/derived/sav_yolo_seg_student/images/train"))
    parser.add_argument("--labels", type=Path, default=Path("data/derived/sav_yolo_seg_student/labels/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sav_student_dataset_preview.jpg"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--thumb-width", type=int, default=320)
    return parser.parse_args()


def draw_label(image: np.ndarray, label_path: Path) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        pts = np.column_stack([coords[:, 0] * w, coords[:, 1] * h]).astype(np.int32)
        color = (80, 210, 110)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.polylines(out, [pts], True, color, 2)
    return out


def main() -> None:
    args = parse_args()
    images = sorted(args.images.glob("*.jpg"))[: args.limit]
    thumbs = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = draw_label(image, args.labels / f"{image_path.stem}.txt")
        scale = args.thumb_width / image.shape[1]
        thumb = cv2.resize(image, (args.thumb_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    if not thumbs:
        raise RuntimeError("No preview images found")
    rows = []
    for idx in range(0, len(thumbs), 3):
        row = thumbs[idx : idx + 3]
        max_h = max(t.shape[0] for t in row)
        padded = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(8, 12, 10)) for t in row]
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(args.output)


if __name__ == "__main__":
    main()
