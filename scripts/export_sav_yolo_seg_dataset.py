from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export SA-V masklets as a YOLO segmentation dataset.")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/sav_subset51_shard_3/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/sav_yolo_seg_student"))
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--val-samples", type=int, default=1)
    parser.add_argument("--min-area-frac", type=float, default=0.001)
    parser.add_argument("--max-objects-per-frame", type=int, default=12)
    parser.add_argument("--contour-epsilon-frac", type=float, default=0.003)
    return parser.parse_args()


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    encoded = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return mask_utils.decode(encoded).astype(np.uint8)


def resize_frame_and_masks(frame: np.ndarray, masks: list[np.ndarray], max_side: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return frame, masks
    size = (round(w * scale), round(h * scale))
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    masks = [cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) for mask in masks]
    return frame, masks


def mask_to_yolo_segments(mask: np.ndarray, epsilon_frac: float) -> list[list[float]]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        epsilon = epsilon_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue
        coords: list[float] = []
        for x, y in approx:
            coords.extend([float(np.clip(x / max(1, w), 0.0, 1.0)), float(np.clip(y / max(1, h), 0.0, 1.0))])
        if len(coords) >= 6:
            segments.append(coords)
    return segments


def write_split_list(paths: list[Path], path: Path) -> None:
    path.write_text("\n".join(str(p.as_posix()) for p in paths) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    out = args.output_dir
    image_root = out / "images"
    label_root = out / "labels"
    for split in ["train", "val"]:
        (image_root / split).mkdir(parents=True, exist_ok=True)
        (label_root / split).mkdir(parents=True, exist_ok=True)

    samples = manifest["samples"][: args.max_samples]
    val_start = max(0, len(samples) - args.val_samples)
    image_paths: dict[str, list[Path]] = {"train": [], "val": []}
    exported_frames = 0
    exported_instances = 0

    for sample_idx, sample in enumerate(samples):
        split = "val" if sample_idx >= val_start else "train"
        cap = cv2.VideoCapture(str(root / sample["video"]))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {root / sample['video']}")
        for frame_idx in range(args.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            raw_objects = sample["frames"].get(str(frame_idx), [])
            masks = [decode_rle(obj["mask_rle"]) for obj in raw_objects]
            frame, masks = resize_frame_and_masks(frame, masks, args.max_side)
            h, w = frame.shape[:2]
            frame_area = float(h * w)
            mask_items = []
            for mask in masks:
                area = float(mask.sum())
                if area / frame_area < args.min_area_frac:
                    continue
                mask_items.append((area, mask))
            mask_items.sort(reverse=True, key=lambda item: item[0])
            label_lines: list[str] = []
            for _, mask in mask_items[: args.max_objects_per_frame]:
                for segment in mask_to_yolo_segments(mask, args.contour_epsilon_frac):
                    label_lines.append("0 " + " ".join(f"{value:.6f}" for value in segment))
            if not label_lines:
                continue
            stem = f"{sample['video_id']}_{frame_idx:05d}"
            image_path = image_root / split / f"{stem}.jpg"
            label_path = label_root / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame)
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            image_paths[split].append(image_path)
            exported_frames += 1
            exported_instances += len(label_lines)
        cap.release()

    write_split_list(image_paths["train"], out / "train.txt")
    write_split_list(image_paths["val"], out / "val.txt")
    yaml = "\n".join(
        [
            f"path: {out.resolve().as_posix()}",
            "train: train.txt",
            "val: val.txt",
            "names:",
            "  0: object",
            "",
        ]
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    summary = {
        "manifest": str(args.manifest),
        "output_dir": str(out),
        "train_frames": len(image_paths["train"]),
        "val_frames": len(image_paths["val"]),
        "instances": exported_instances,
        "max_side": args.max_side,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
