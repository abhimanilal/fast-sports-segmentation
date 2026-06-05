from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from track_segment_video import (  # noqa: E402
    LocateAnythingDetector,
    resolve_dtype as resolve_model_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed EdgeTAM video tracking from YOLO person boxes on frame 0."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edgetam-root", type=Path, default=Path("vendor/EdgeTAM"))
    parser.add_argument(
        "--edgetam-checkpoint",
        type=Path,
        default=Path("vendor/EdgeTAM/checkpoints/edgetam.pt"),
    )
    parser.add_argument("--edgetam-config", default="configs/edgetam.yaml")
    parser.add_argument("--seed-detector", choices=["yolo", "locate"], default="yolo")
    parser.add_argument("--yolo-model", default="yolo26n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=480)
    parser.add_argument("--locate-model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--locate-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--locate-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--locate-load-in-8bit", action="store_true")
    parser.add_argument("--locate-max-gpu-memory", default="5200MiB")
    parser.add_argument("--locate-max-cpu-memory", default="8GiB")
    parser.add_argument("--locate-offload-folder", default="E:/HFOffload/LocateAnything8bit")
    parser.add_argument("--categories", nargs="+", default=["basketball player"])
    parser.add_argument("--generation-mode", default="fast", choices=["fast", "hybrid", "slow"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--min-box-area-frac", type=float, default=0.0015)
    parser.add_argument("--max-box-area-frac", type=float, default=0.22)
    parser.add_argument("--min-box-side", type=float, default=8.0)
    parser.add_argument("--max-box-aspect", type=float, default=5.0)
    parser.add_argument("--edge-margin-frac", type=float, default=0.01)
    parser.add_argument("--reject-edge-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))))


def extract_frames(video: Path, frames_dir: Path, max_frames: int, max_side: int) -> tuple[list[np.ndarray], float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.jpg"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        resized = resize_max_side(frame_bgr, max_side)
        frames.append(resized)
        cv2.imwrite(str(frames_dir / f"{len(frames) - 1:05d}.jpg"), resized)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video}")
    return frames, fps


def is_usable_box(
    box: list[float],
    width: int,
    height: int,
    min_area_frac: float,
    max_area_frac: float,
    min_side: float,
    max_aspect: float,
    edge_margin_frac: float,
    reject_edge: bool,
) -> bool:
    x1, y1, x2, y2 = box
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    area = box_w * box_h
    image_area = float(width * height)
    if box_w < min_side or box_h < min_side:
        return False
    if area < image_area * min_area_frac or area > image_area * max_area_frac:
        return False
    aspect = max(box_w / max(1.0, box_h), box_h / max(1.0, box_w))
    if aspect > max_aspect:
        return False
    edge_margin = min(width, height) * edge_margin_frac
    if reject_edge and (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= width - 1 - edge_margin
        or y2 >= height - 1 - edge_margin
    ):
        return False
    return True


def detect_seed_boxes(
    frame_bgr: np.ndarray,
    model_name: str,
    conf: float,
    imgsz: int,
    max_objects: int,
    min_area_frac: float,
    max_area_frac: float,
    min_side: float,
    max_aspect: float,
    edge_margin_frac: float,
    reject_edge: bool,
) -> tuple[list[list[float]], dict[str, int]]:
    from ultralytics import YOLO

    model = YOLO(model_name)
    result = model.predict(frame_bgr[:, :, ::-1], conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes: list[tuple[float, list[float]]] = []
    stats = {"raw": 0, "class": 0, "filtered": 0, "kept": 0}
    if result.boxes is None:
        return [], stats
    height, width = frame_bgr.shape[:2]
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy()
    score = result.boxes.conf.detach().cpu().numpy()
    for box, class_id, confidence in zip(xyxy, cls, score):
        stats["raw"] += 1
        if int(class_id) != 0:
            continue
        stats["class"] += 1
        x1, y1, x2, y2 = [float(v) for v in box]
        candidate = [x1, y1, x2, y2]
        if not is_usable_box(
            candidate,
            width,
            height,
            min_area_frac,
            max_area_frac,
            min_side,
            max_aspect,
            edge_margin_frac,
            reject_edge,
        ):
            stats["filtered"] += 1
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        boxes.append((float(confidence) * area, candidate))
    boxes.sort(key=lambda item: item[0], reverse=True)
    kept = [box for _, box in boxes[:max_objects]]
    stats["kept"] = len(kept)
    return kept, stats


def filter_seed_boxes(
    boxes: list[list[float]],
    frame_bgr: np.ndarray,
    max_objects: int,
    min_area_frac: float,
    max_area_frac: float,
    min_side: float,
    max_aspect: float,
    edge_margin_frac: float,
    reject_edge: bool,
) -> tuple[list[list[float]], dict[str, int]]:
    height, width = frame_bgr.shape[:2]
    kept_with_area: list[tuple[float, list[float]]] = []
    stats = {"raw": len(boxes), "class": len(boxes), "filtered": 0, "kept": 0}
    for box in boxes:
        candidate = [float(v) for v in box]
        if not is_usable_box(
            candidate,
            width,
            height,
            min_area_frac,
            max_area_frac,
            min_side,
            max_aspect,
            edge_margin_frac,
            reject_edge,
        ):
            stats["filtered"] += 1
            continue
        x1, y1, x2, y2 = candidate
        kept_with_area.append((max(0.0, x2 - x1) * max(0.0, y2 - y1), candidate))
    kept_with_area.sort(key=lambda item: item[0], reverse=True)
    kept = [box for _, box in kept_with_area[:max_objects]]
    stats["kept"] = len(kept)
    return kept, stats


def locate_seed_boxes(
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[list[float]], dict[str, int], str]:
    locate_dtype = resolve_model_dtype(args.locate_dtype, args.locate_device)
    detector = LocateAnythingDetector(
        args.locate_model,
        args.locate_device,
        locate_dtype,
        args.locate_load_in_8bit,
        args.locate_max_gpu_memory,
        args.locate_max_cpu_memory,
        args.locate_offload_folder,
    )
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    answer, boxes = detector.detect(
        image,
        args.categories,
        args.generation_mode,
        args.max_new_tokens,
    )
    kept, stats = filter_seed_boxes(
        boxes,
        frame_bgr,
        args.max_objects,
        args.min_box_area_frac,
        args.max_box_area_frac,
        args.min_box_side,
        args.max_box_aspect,
        args.edge_margin_frac,
        args.reject_edge_seeds,
    )
    return kept, stats, answer


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def overlay_masks(frame_bgr: np.ndarray, masks_by_id: dict[int, np.ndarray]) -> np.ndarray:
    colors = [
        (25, 118, 210),
        (216, 67, 21),
        (46, 125, 50),
        (251, 192, 45),
        (123, 31, 162),
        (0, 137, 123),
        (198, 40, 40),
        (94, 53, 177),
    ]
    output = frame_bgr.copy()
    for obj_id, mask in masks_by_id.items():
        color = colors[obj_id % len(colors)]
        output[mask] = (0.55 * output[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
        box = mask_to_box(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"id={obj_id}",
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "edgetam_frames"

    frames, source_fps = extract_frames(args.video, frames_dir, args.max_frames, args.max_side)
    seed_start = time.perf_counter()
    seed_answer = None
    if args.seed_detector == "yolo":
        seed_boxes, seed_filter_stats = detect_seed_boxes(
            frames[0],
            args.yolo_model,
            args.yolo_conf,
            args.yolo_imgsz,
            args.max_objects,
            args.min_box_area_frac,
            args.max_box_area_frac,
            args.min_box_side,
            args.max_box_aspect,
            args.edge_margin_frac,
            args.reject_edge_seeds,
        )
    else:
        seed_boxes, seed_filter_stats, seed_answer = locate_seed_boxes(frames[0], args)
    seed_seconds = time.perf_counter() - seed_start
    if not seed_boxes:
        raise RuntimeError(f"{args.seed_detector} found no usable seed boxes on the first extracted frame")

    edge_root = args.edgetam_root.resolve()
    if str(edge_root) not in sys.path:
        sys.path.insert(0, str(edge_root))
    from sam2.build_sam import build_sam2_video_predictor

    dtype = resolve_dtype(args.dtype)
    autocast_enabled = args.device.startswith("cuda") and dtype != torch.float32
    start_build = time.perf_counter()
    predictor = build_sam2_video_predictor(
        args.edgetam_config,
        str(args.edgetam_checkpoint),
        device=args.device,
    )
    build_seconds = time.perf_counter() - start_build

    metrics: dict[str, Any] = {
        "video": str(args.video),
        "source_fps": source_fps,
        "frames": [],
        "seed_boxes": seed_boxes,
        "seed_filter_stats": seed_filter_stats,
        "seed_detector": args.seed_detector,
        "seed_seconds": seed_seconds,
        "seed_answer": seed_answer,
        "yolo_model": args.yolo_model,
        "locate_model": args.locate_model if args.seed_detector == "locate" else None,
        "categories": args.categories if args.seed_detector == "locate" else None,
        "edgetam_checkpoint": str(args.edgetam_checkpoint),
        "build_seconds": build_seconds,
        "dtype": args.dtype,
    }

    writer = None
    if args.write_video:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(args.output_dir / "tracked.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (w, h),
        )

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=dtype, enabled=autocast_enabled
    ):
        start_init = time.perf_counter()
        state = predictor.init_state(
            str(frames_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )
        metrics["init_seconds"] = time.perf_counter() - start_init

        for obj_id, box in enumerate(seed_boxes):
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=np.asarray(box))

        start_propagate = time.perf_counter()
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, max_frame_num_to_track=len(frames)
        ):
            frame_start = time.perf_counter()
            masks_tensor = (mask_logits > 0.0).detach().cpu()
            masks_by_id: dict[int, np.ndarray] = {}
            boxes: list[list[float]] = []
            areas: list[int] = []
            for mask_idx, obj_id in enumerate(obj_ids):
                mask = masks_tensor[mask_idx, 0].numpy().astype(bool)
                masks_by_id[int(obj_id)] = mask
                box = mask_to_box(mask)
                boxes.append(box or [])
                areas.append(int(mask.sum()))
            if writer is not None and frame_idx < len(frames):
                writer.write(overlay_masks(frames[frame_idx], masks_by_id))
            metrics["frames"].append(
                {
                    "frame": int(frame_idx),
                    "object_ids": [int(obj_id) for obj_id in obj_ids],
                    "boxes": boxes,
                    "mask_areas": areas,
                    "frame_seconds": time.perf_counter() - frame_start,
                }
            )
        metrics["propagate_seconds"] = time.perf_counter() - start_propagate

    if writer is not None:
        writer.release()
    processed = max(1, len(metrics["frames"]))
    metrics["propagate_fps"] = processed / max(1e-9, metrics["propagate_seconds"])
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"seeded {len(seed_boxes)} objects; propagated {processed} frames at "
        f"{metrics['propagate_fps']:.2f} FPS; wrote {args.output_dir}"
    )


if __name__ == "__main__":
    main()
