from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small SA-V Subset 51 shard from Hugging Face/FiftyOne and export videos plus RLE annotations."
    )
    parser.add_argument("--repo-id", default="Voxel51/segment_anything_video_subset51")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/sav_subset51"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/sav_subset51_shard"))
    parser.add_argument("--annotation-field", choices=["manual", "auto"], default="manual")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def video_meta(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {path}")
    height, width = frame.shape[:2]
    meta = {
        "width": width,
        "height": height,
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return meta


def full_mask_from_detection(det, width: int, height: int) -> np.ndarray:
    x, y, w, h = [float(v) for v in det.bounding_box]
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + w) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + h) * height))))
    full = np.zeros((height, width), dtype=np.uint8)
    if det.mask is None:
        return full
    crop = np.asarray(det.mask).astype(np.uint8)
    if crop.shape[:2] != (y2 - y1, x2 - x1):
        crop = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full[y1:y2, x1:x2] = crop
    return full


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(v) for v in rle["size"]],
        "counts": rle["counts"].decode("ascii"),
    }


def box_from_mask(binary: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(binary)
    if xs.size == 0 or ys.size == 0:
        return []
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    from fiftyone.utils.huggingface import load_from_hub

    dataset = load_from_hub(
        args.repo_id,
        max_samples=args.max_samples,
        dataset_dir=str(args.dataset_dir),
        overwrite=args.overwrite,
    )
    samples = []
    for sample in dataset:
        source_video = Path(sample.filepath)
        target_video = videos_dir / source_video.name
        if not target_video.exists() or args.overwrite:
            shutil.copy2(source_video, target_video)
        meta = video_meta(target_video)
        frames: dict[str, Any] = {}
        for frame_number, frame in sample.frames.items():
            if args.annotation_field not in frame.field_names:
                continue
            detections = frame[args.annotation_field]
            if detections is None:
                continue
            objects = []
            for det in detections.detections:
                binary = full_mask_from_detection(det, meta["width"], meta["height"])
                objects.append(
                    {
                        "masklet_id": str(det.masklet_id),
                        "bbox_xyxy": box_from_mask(binary),
                        "mask_rle": encode_binary_mask(binary),
                        "masklet_size_bucket": getattr(det, "masklet_size_bucket", None),
                        "masklet_size_rel": getattr(det, "masklet_size_rel", None),
                    }
                )
            if objects:
                frames[str(int(frame_number) - 1)] = objects
        samples.append(
            {
                "video_id": sample.video_id,
                "video": str(target_video.relative_to(args.output_dir)),
                "source_filepath": sample.filepath,
                "annotation_field": args.annotation_field,
                "metadata": meta,
                "frames": frames,
            }
        )

    manifest = {
        "source": args.repo_id,
        "annotation_field": args.annotation_field,
        "samples": samples,
    }
    out_path = args.output_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
