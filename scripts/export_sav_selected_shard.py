from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
import ijson
import numpy as np
from huggingface_hub import hf_hub_download
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export selected SA-V videos and manual masklets to the repo manifest format.")
    parser.add_argument("--repo-id", default="Voxel51/segment_anything_video_subset51")
    parser.add_argument(
        "--hub-root",
        type=Path,
        default=Path(r"E:\FiftyOne\huggingface\hub\Voxel51\segment_anything_video_subset51"),
    )
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/sav_sports_shard"))
    parser.add_argument("--annotation-field", choices=["manual", "auto"], default="manual")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def clean_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    return value


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


def full_mask_from_detection(det: fo.Detection, width: int, height: int) -> np.ndarray:
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
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}


def box_from_mask(binary: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(binary)
    if xs.size == 0 or ys.size == 0:
        return []
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def download_or_copy_video(repo_id: str, sample: dict[str, Any], hub_root: Path, videos_dir: Path, overwrite: bool) -> Path:
    source = hub_root / sample["filepath"]
    if not source.exists():
        source = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=sample["filepath"].replace("\\", "/"), local_dir=hub_root))
    target = videos_dir / Path(sample["filepath"]).name
    if overwrite or not target.exists():
        shutil.copy2(source, target)
    return target


def main() -> None:
    args = parse_args()
    selected_ids = {line.strip() for line in args.selected_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    samples = json.loads((args.hub_root / "samples.json").read_text(encoding="utf-8"))["samples"]
    selected_samples = [sample for sample in samples if sample["video_id"] in selected_ids]
    if not selected_samples:
        raise RuntimeError("No selected sample IDs matched samples.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    sample_by_oid = {sample["_id"]["$oid"]: sample for sample in selected_samples}
    exported: dict[str, dict[str, Any]] = {}
    for sample in selected_samples:
        video_path = download_or_copy_video(args.repo_id, sample, args.hub_root, videos_dir, args.overwrite)
        meta = video_meta(video_path)
        exported[sample["_id"]["$oid"]] = {
            "video_id": sample["video_id"],
            "video": str(video_path.relative_to(args.output_dir)),
            "source_filepath": sample["filepath"],
            "annotation_field": args.annotation_field,
            "metadata": meta,
            "frames": {},
        }

    frames_path = args.hub_root / "frames.json"
    matched_frames = 0
    with frames_path.open("rb") as stream:
        for frame_doc in ijson.items(stream, "frames.item"):
            sample_id = frame_doc.get("_sample_id", {}).get("$oid")
            if sample_id not in sample_by_oid:
                continue
            label_doc = frame_doc.get(args.annotation_field)
            if not label_doc:
                continue
            detections = label_doc.get("detections") or []
            if not detections:
                continue
            sample_out = exported[sample_id]
            meta = sample_out["metadata"]
            objects = []
            for det_doc in detections:
                det = fo.Detection.from_dict(clean_json(det_doc))
                binary = full_mask_from_detection(det, meta["width"], meta["height"])
                bbox = box_from_mask(binary)
                if not bbox:
                    continue
                objects.append(
                    {
                        "masklet_id": str(det.masklet_id),
                        "bbox_xyxy": bbox,
                        "mask_rle": encode_binary_mask(binary),
                        "masklet_size_bucket": getattr(det, "masklet_size_bucket", None),
                        "masklet_size_rel": getattr(det, "masklet_size_rel", None),
                    }
                )
            if objects:
                sample_out["frames"][str(int(frame_doc["frame_number"]) - 1)] = objects
                matched_frames += 1

    manifest = {
        "source": args.repo_id,
        "annotation_field": args.annotation_field,
        "selection": sorted(selected_ids),
        "samples": list(exported.values()),
    }
    out_path = args.output_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(exported)} samples and {matched_frames} annotated frames to {out_path}")


if __name__ == "__main__":
    main()
