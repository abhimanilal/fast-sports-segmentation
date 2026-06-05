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
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_sav_edgetam import (  # noqa: E402
    binary_iou,
    box_iou,
    mask_to_box,
    overlay,
    read_sample_frames,
    select_seed_ids,
)
from track_segment_video import (  # noqa: E402
    Track,
    crop_gray,
    init_tracks,
    resolve_dtype,
    sam_segment,
    update_track,
    update_tracks_from_masks,
)

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_efficientsam3_image_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientSAM sparse mask refresh tracking on a local SA-V shard."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/sav_subset51_shard/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sav_efficientsam_benchmark"))
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("models/efficient_sam3_efficientvit_s_point_prompt_slim.pt"),
    )
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--sam-every", type=int, default=5)
    parser.add_argument("--search-pad", type=float, default=1.6)
    parser.add_argument("--template-update", type=float, default=0.08)
    parser.add_argument("--mask-update-threshold", type=float, default=0.55)
    parser.add_argument("--mask-box-pad-frac", type=float, default=0.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def build_seed_tracks(
    first_frame: np.ndarray,
    first_masks: dict[str, np.ndarray],
    seed_ids: list[str],
) -> tuple[list[Track], dict[int, str]]:
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = first_frame.shape[:2]
    boxes = [mask_to_box(first_masks[object_id]) for object_id in seed_ids]
    tracks, _ = init_tracks(gray, boxes, w, h, start_track_id=0)
    object_id_map = {idx: object_id for idx, object_id in enumerate(seed_ids)}
    return tracks, object_id_map


def score_tracks(
    video_id: str,
    frame_idx: int,
    tracks: list[Track],
    object_id_map: dict[int, str],
    gt_masks: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    box_ious: list[float],
    tracked_frames_by_id: defaultdict[str, int],
) -> None:
    for track in tracks:
        object_id = object_id_map.get(track.track_id)
        if object_id is None:
            continue
        gt_mask = gt_masks.get(object_id)
        if gt_mask is None:
            continue
        gt_box = mask_to_box(gt_mask)
        biou = box_iou(track.box.astype(float).tolist(), gt_box)
        box_ious.append(biou)
        tracked_frames_by_id[object_id] += 1
        rows.append(
            {
                "video_id": video_id,
                "frame": frame_idx,
                "masklet_id": object_id,
                "metric_type": "box",
                "mask_iou": "",
                "box_iou": biou,
                "gt_area": int(gt_mask.sum()),
                "pred_area": "",
            }
        )


def score_masks(
    video_id: str,
    frame_idx: int,
    tracks: list[Track],
    object_id_map: dict[int, str],
    masks: np.ndarray,
    gt_masks: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    mask_ious: list[float],
) -> dict[str, float]:
    frame_scores: dict[str, float] = {}
    for idx, track in enumerate(tracks):
        object_id = object_id_map.get(track.track_id)
        if object_id is None or idx >= len(masks):
            continue
        gt_mask = gt_masks.get(object_id)
        if gt_mask is None:
            continue
        pred_mask = np.squeeze(masks[idx]).astype(bool)
        miou = binary_iou(pred_mask, gt_mask)
        biou = box_iou(mask_to_box(pred_mask), mask_to_box(gt_mask))
        frame_scores[object_id] = miou
        mask_ious.append(miou)
        rows.append(
            {
                "video_id": video_id,
                "frame": frame_idx,
                "masklet_id": object_id,
                "metric_type": "mask_refresh",
                "mask_iou": miou,
                "box_iou": biou,
                "gt_area": int(gt_mask.sum()),
                "pred_area": int(pred_mask.sum()),
            }
        )
    return frame_scores


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shard_root = args.manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype, args.device)
    model = build_efficientsam3_image_model(
        checkpoint_path=str(args.sam_checkpoint),
        backbone_type="efficientvit",
        model_name="b0",
        enable_text_encoder=False,
        enable_inst_interactivity=True,
        device=args.device,
    )
    if args.device == "cuda" and dtype != torch.float32:
        model = model.to(dtype=dtype)
    processor = Sam3Processor(model, device=args.device)

    rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    total_seconds = 0.0
    total_frames = 0
    total_sam_seconds = 0.0
    total_sam_calls = 0

    for sample in manifest["samples"][: args.max_samples]:
        video_id = sample["video_id"]
        sample_dir = args.output_dir / video_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        frames, gt_by_frame, _ = read_sample_frames(sample, shard_root, args.max_frames, args.max_side)
        if not frames:
            continue
        seed_ids = select_seed_ids(gt_by_frame, args.max_objects)
        if not seed_ids:
            sample_summaries.append({"video_id": video_id, "skipped": "no frame-0 masklets"})
            continue

        tracks, object_id_map = build_seed_tracks(frames[0], gt_by_frame[0], seed_ids)
        writer = None
        if args.write_video:
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(sample_dir / "overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(sample["metadata"].get("fps") or 6.0),
                (w, h),
            )

        mask_ious: list[float] = []
        box_ious: list[float] = []
        tracked_frames_by_id: defaultdict[str, int] = defaultdict(int)
        sample_sam_seconds = 0.0
        sample_sam_calls = 0
        start = time.perf_counter()
        last_masks = None
        last_scores: dict[str, float] = {}

        for frame_idx, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_idx > 0:
                for track in tracks:
                    update_track(gray, track, args.search_pad, args.template_update)

            should_refresh = frame_idx % args.sam_every == 0
            if tracks and should_refresh:
                boxes = np.asarray([track.box for track in tracks], dtype=np.float32)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                masks, scores, sam_seconds = sam_segment(
                    model, processor, image, boxes, args.device, dtype
                )
                sample_sam_seconds += sam_seconds
                sample_sam_calls += 1
                update_tracks_from_masks(
                    gray,
                    tracks,
                    masks,
                    scores,
                    args.mask_update_threshold,
                    args.mask_box_pad_frac,
                )
                last_masks = masks
                last_scores = score_masks(
                    video_id,
                    frame_idx,
                    tracks,
                    object_id_map,
                    masks,
                    gt_by_frame.get(frame_idx, {}),
                    rows,
                    mask_ious,
                )

            score_tracks(
                video_id,
                frame_idx,
                tracks,
                object_id_map,
                gt_by_frame.get(frame_idx, {}),
                rows,
                box_ious,
                tracked_frames_by_id,
            )
            if writer is not None:
                pred_masks = {}
                if last_masks is not None:
                    for idx, track in enumerate(tracks):
                        object_id = object_id_map.get(track.track_id)
                        if object_id is not None and idx < len(last_masks):
                            pred_masks[object_id] = np.squeeze(last_masks[idx]).astype(bool)
                writer.write(overlay(frames[frame_idx], pred_masks, last_scores))

        elapsed = time.perf_counter() - start
        if writer is not None:
            writer.release()
        total_seconds += elapsed
        total_frames += len(frames)
        total_sam_seconds += sample_sam_seconds
        total_sam_calls += sample_sam_calls
        sample_summaries.append(
            {
                "video_id": video_id,
                "seed_ids": seed_ids,
                "frames": len(frames),
                "mask_refresh_pairs": len(mask_ious),
                "box_pairs": len(box_ious),
                "mean_mask_iou_refresh": float(np.mean(mask_ious)) if mask_ious else 0.0,
                "mean_box_iou_all": float(np.mean(box_ious)) if box_ious else 0.0,
                "mask_j_at_50_refresh": float(np.mean([iou >= 0.5 for iou in mask_ious])) if mask_ious else 0.0,
                "box_j_at_50_all": float(np.mean([iou >= 0.5 for iou in box_ious])) if box_ious else 0.0,
                "seconds": elapsed,
                "fps": len(frames) / max(1e-9, elapsed),
                "sam_seconds": sample_sam_seconds,
                "sam_calls": sample_sam_calls,
                "tracked_frames_by_id": dict(tracked_frames_by_id),
            }
        )

    rows_path = args.output_dir / "per_frame_metrics.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame",
                "masklet_id",
                "metric_type",
                "mask_iou",
                "box_iou",
                "gt_area",
                "pred_area",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mask_ious_all = [float(row["mask_iou"]) for row in rows if row["metric_type"] == "mask_refresh"]
    box_ious_all = [float(row["box_iou"]) for row in rows if row["metric_type"] == "box"]
    summary = {
        "manifest": str(args.manifest),
        "system": "efficientsam_oracle_first_frame_box_sparse_refresh",
        "sam_every": args.sam_every,
        "samples": sample_summaries,
        "mask_refresh_pairs": len(mask_ious_all),
        "box_pairs": len(box_ious_all),
        "mean_mask_iou_refresh": float(np.mean(mask_ious_all)) if mask_ious_all else 0.0,
        "mean_box_iou_all": float(np.mean(box_ious_all)) if box_ious_all else 0.0,
        "mask_j_at_50_refresh": float(np.mean([iou >= 0.5 for iou in mask_ious_all])) if mask_ious_all else 0.0,
        "box_j_at_50_all": float(np.mean([iou >= 0.5 for iou in box_ious_all])) if box_ious_all else 0.0,
        "seconds": total_seconds,
        "fps": total_frames / max(1e-9, total_seconds),
        "sam_seconds": total_sam_seconds,
        "sam_calls": total_sam_calls,
        "sam_seconds_avg": total_sam_seconds / max(1, total_sam_calls),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"sam_every={args.sam_every}; mask refresh pairs={summary['mask_refresh_pairs']}; "
        f"mean refresh mask IoU={summary['mean_mask_iou_refresh']:.3f}; "
        f"box pairs={summary['box_pairs']}; mean all-frame box IoU={summary['mean_box_iou_all']:.3f}; "
        f"FPS={summary['fps']:.2f}"
    )


if __name__ == "__main__":
    main()
