from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EdgeTAM video tracking on a local SA-V shard with oracle first-frame box prompts."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/sav_subset51_shard/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sav_edgetam_benchmark"))
    parser.add_argument("--edgetam-root", type=Path, default=Path("vendor/EdgeTAM"))
    parser.add_argument("--edgetam-checkpoint", type=Path, default=Path("vendor/EdgeTAM/checkpoints/edgetam.pt"))
    parser.add_argument("--edgetam-config", default="configs/edgetam.yaml")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    encoded = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return mask_utils.decode(encoded).astype(bool)


def resize_frame_and_mask(
    frame: np.ndarray,
    masks_by_id: dict[str, np.ndarray],
    max_side: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    if max_side <= 0:
        return frame, masks_by_id, 1.0
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame, masks_by_id, 1.0
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized_frame = cv2.resize(frame, new_size)
    resized_masks = {
        object_id: cv2.resize(mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        for object_id, mask in masks_by_id.items()
    }
    return resized_frame, resized_masks, scale


def mask_to_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return []
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    if not box_a or not box_b:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def read_sample_frames(sample: dict[str, Any], root: Path, max_frames: int, max_side: int):
    video_path = root / sample["video"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frames: list[np.ndarray] = []
    gt_by_frame: dict[int, dict[str, np.ndarray]] = {}
    scales: dict[int, float] = {}
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        raw_objects = sample["frames"].get(str(frame_idx), [])
        masks = {obj["masklet_id"]: decode_rle(obj["mask_rle"]) for obj in raw_objects}
        frame, masks, scale = resize_frame_and_mask(frame, masks, max_side)
        frames.append(frame)
        gt_by_frame[frame_idx] = masks
        scales[frame_idx] = scale
        frame_idx += 1
    cap.release()
    return frames, gt_by_frame, scales


def write_frame_folder(frames: list[np.ndarray], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"{idx:05d}.jpg"), frame)


def select_seed_ids(gt_by_frame: dict[int, dict[str, np.ndarray]], max_objects: int) -> list[str]:
    first_masks = gt_by_frame.get(0, {})
    candidates = [(int(mask.sum()), object_id) for object_id, mask in first_masks.items()]
    candidates.sort(reverse=True)
    return [object_id for _, object_id in candidates[:max_objects]]


def overlay(frame: np.ndarray, masks: dict[str, np.ndarray], scores: dict[str, float]) -> np.ndarray:
    colors = [(25, 118, 210), (216, 67, 21), (46, 125, 50), (251, 192, 45)]
    out = frame.copy()
    for idx, (object_id, mask) in enumerate(masks.items()):
        color = colors[idx % len(colors)]
        out[mask] = (0.55 * out[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
        box = mask_to_box(mask)
        if box:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"{object_id} iou={scores.get(object_id, 0.0):.2f}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shard_root = args.manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edge_root = args.edgetam_root.resolve()
    if str(edge_root) not in sys.path:
        sys.path.insert(0, str(edge_root))
    from sam2.build_sam import build_sam2_video_predictor

    dtype = resolve_dtype(args.dtype)
    autocast_enabled = args.device.startswith("cuda") and dtype != torch.float32
    predictor = build_sam2_video_predictor(
        args.edgetam_config,
        str(args.edgetam_checkpoint),
        device=args.device,
    )

    rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    total_propagate_seconds = 0.0
    total_predicted_frames = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        for sample in manifest["samples"][: args.max_samples]:
            video_id = sample["video_id"]
            sample_dir = args.output_dir / video_id
            frame_dir = sample_dir / "frames"
            frames, gt_by_frame, _ = read_sample_frames(sample, shard_root, args.max_frames, args.max_side)
            if not frames:
                continue
            seed_ids = select_seed_ids(gt_by_frame, args.max_objects)
            if not seed_ids:
                sample_summaries.append({"video_id": video_id, "skipped": "no frame-0 masklets"})
                continue
            write_frame_folder(frames, frame_dir)
            state = predictor.init_state(
                str(frame_dir),
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
            object_id_map = {idx: object_id for idx, object_id in enumerate(seed_ids)}
            for idx, object_id in object_id_map.items():
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=idx,
                    box=np.asarray(mask_to_box(gt_by_frame[0][object_id]), dtype=np.float32),
                )

            writer = None
            if args.write_video:
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(sample_dir / "overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(sample["metadata"].get("fps") or 6.0),
                    (w, h),
                )

            start = time.perf_counter()
            per_sample_ious: list[float] = []
            per_sample_box_ious: list[float] = []
            tracked_frames_by_id: defaultdict[str, int] = defaultdict(int)
            for frame_idx, edge_obj_ids, mask_logits in predictor.propagate_in_video(
                state, max_frame_num_to_track=len(frames)
            ):
                pred_masks_by_id: dict[str, np.ndarray] = {}
                frame_scores: dict[str, float] = {}
                masks_tensor = (mask_logits > 0.0).detach().cpu()
                for mask_idx, edge_obj_id in enumerate(edge_obj_ids):
                    object_id = object_id_map[int(edge_obj_id)]
                    pred_mask = masks_tensor[mask_idx, 0].numpy().astype(bool)
                    pred_masks_by_id[object_id] = pred_mask
                    gt_mask = gt_by_frame.get(int(frame_idx), {}).get(object_id)
                    if gt_mask is None:
                        continue
                    miou = binary_iou(pred_mask, gt_mask)
                    biou = box_iou(mask_to_box(pred_mask), mask_to_box(gt_mask))
                    frame_scores[object_id] = miou
                    per_sample_ious.append(miou)
                    per_sample_box_ious.append(biou)
                    tracked_frames_by_id[object_id] += 1
                    rows.append(
                        {
                            "video_id": video_id,
                            "frame": int(frame_idx),
                            "masklet_id": object_id,
                            "mask_iou": miou,
                            "box_iou": biou,
                            "gt_area": int(gt_mask.sum()),
                            "pred_area": int(pred_mask.sum()),
                        }
                    )
                if writer is not None:
                    writer.write(overlay(frames[int(frame_idx)], pred_masks_by_id, frame_scores))
            propagate_seconds = time.perf_counter() - start
            if writer is not None:
                writer.release()

            total_propagate_seconds += propagate_seconds
            total_predicted_frames += len(frames)
            sample_summaries.append(
                {
                    "video_id": video_id,
                    "seed_ids": seed_ids,
                    "frames": len(frames),
                    "evaluated_pairs": len(per_sample_ious),
                    "mean_mask_iou": float(np.mean(per_sample_ious)) if per_sample_ious else 0.0,
                    "mean_box_iou": float(np.mean(per_sample_box_ious)) if per_sample_box_ious else 0.0,
                    "j_at_50": float(np.mean([iou >= 0.5 for iou in per_sample_ious])) if per_sample_ious else 0.0,
                    "propagate_seconds": propagate_seconds,
                    "propagate_fps": len(frames) / max(1e-9, propagate_seconds),
                    "tracked_frames_by_id": dict(tracked_frames_by_id),
                }
            )

    rows_path = args.output_dir / "per_frame_metrics.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "frame", "masklet_id", "mask_iou", "box_iou", "gt_area", "pred_area"],
        )
        writer.writeheader()
        writer.writerows(rows)

    all_ious = [float(row["mask_iou"]) for row in rows]
    all_box_ious = [float(row["box_iou"]) for row in rows]
    summary = {
        "manifest": str(args.manifest),
        "system": "edgetam_oracle_first_frame_box",
        "samples": sample_summaries,
        "evaluated_pairs": len(rows),
        "mean_mask_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_box_iou": float(np.mean(all_box_ious)) if all_box_ious else 0.0,
        "j_at_50": float(np.mean([iou >= 0.5 for iou in all_ious])) if all_ious else 0.0,
        "propagate_seconds": total_propagate_seconds,
        "propagate_fps": total_predicted_frames / max(1e-9, total_propagate_seconds),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"evaluated {len(rows)} object-frame pairs; "
        f"mean mask IoU={summary['mean_mask_iou']:.3f}; "
        f"J@0.5={summary['j_at_50']:.3f}; "
        f"propagate FPS={summary['propagate_fps']:.2f}"
    )


if __name__ == "__main__":
    main()
