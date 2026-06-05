from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment sampled sports-video frames with an EfficientSAM3 point prompt."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation"))
    parser.add_argument("--stride", type=int, default=48, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--point-x-frac", type=float, default=0.492)
    parser.add_argument("--point-y-frac", type=float, default=0.509)
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--text-context-length", type=int, default=77)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="fp16" if torch.cuda.is_available() else "fp32",
        choices=["fp16", "bf16", "fp32"],
        help="Model/inference dtype. FP16 is the default CUDA memory-saving path.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def autocast_context(device: str, dtype: torch.dtype):
    enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def resize_rgb(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame_rgb, scale


def overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame_rgb.copy()
    color = np.array([245, 90, 40], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    return overlay


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype, args.device)

    model = build_efficientsam3_image_model(
        checkpoint_path=str(args.checkpoint),
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        text_encoder_context_length=args.text_context_length,
        enable_text_encoder=False,
        enable_inst_interactivity=True,
        device=args.device,
    )
    if args.device == "cuda" and dtype != torch.float32:
        model = model.to(dtype=dtype)
    processor = Sam3Processor(model, device=args.device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timings: list[dict[str, float | int | str]] = []
    processed = 0
    frame_index = 0
    warmup_done = False

    while processed < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb, scale = resize_rgb(frame_bgr, args.max_side)
        h, w = frame_rgb.shape[:2]
        point_coords = np.array(
            [[args.point_x_frac * w, args.point_y_frac * h]], dtype=np.float32
        )
        point_labels = np.array([1], dtype=np.int32)

        image = Image.fromarray(frame_rgb)
        start = time.perf_counter()
        with torch.inference_mode(), autocast_context(args.device, dtype):
            state = processor.set_image(image)
            masks, scores, _ = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        scores_np = (
            scores.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor)
            else np.asarray(scores)
        )
        masks_np = (
            masks.detach().cpu().numpy()
            if isinstance(masks, torch.Tensor)
            else np.asarray(masks)
        )
        best_idx = int(np.argmax(scores_np))
        best_mask = np.squeeze(masks_np[best_idx]).astype(bool)
        best_score = float(scores_np[best_idx])
        mask_pixels = int(best_mask.sum())

        vis = overlay_mask(frame_rgb, best_mask)
        cv2.circle(
            vis,
            (int(point_coords[0, 0]), int(point_coords[0, 1])),
            7,
            (255, 255, 255),
            -1,
        )
        out_path = args.output_dir / f"frame_{frame_index:05d}_mask.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        timings.append(
            {
                "frame": frame_index,
                "output": str(out_path),
                "seconds": elapsed,
                "score": best_score,
                "mask_pixels": mask_pixels,
                "scale": scale,
            }
        )
        warmup_done = warmup_done or processed == 0
        processed += 1
        frame_index += args.stride

    cap.release()

    measured = timings[1:] if len(timings) > 1 else timings
    avg_seconds = float(np.mean([item["seconds"] for item in measured])) if measured else 0.0
    metrics = {
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "dtype": str(dtype).replace("torch.", ""),
        "fps": fps,
        "total_frames": total_frames,
        "original_size": [original_w, original_h],
        "processed_frames": processed,
        "avg_seconds_excluding_first": avg_seconds,
        "effective_fps_excluding_first": (1.0 / avg_seconds) if avg_seconds else 0.0,
        "frames": timings,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
