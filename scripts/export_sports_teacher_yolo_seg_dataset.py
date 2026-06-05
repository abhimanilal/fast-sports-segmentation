from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export sports pseudo-labels from a stronger YOLO-Seg teacher.")
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/sports_teacher_yolo_seg"))
    parser.add_argument("--teacher", default="yolo11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.28)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-frames-per-video", type=int, default=180)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-objects", type=int, default=12)
    parser.add_argument("--min-area-frac", type=float, default=0.0015)
    parser.add_argument("--max-area-frac", type=float, default=0.25)
    parser.add_argument("--contour-epsilon-frac", type=float, default=0.003)
    parser.add_argument(
        "--roi-json",
        type=Path,
        help="Optional JSON mapping video stem to normalized active-play polygon points.",
    )
    return parser.parse_args()


def resize_max_side(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return frame, 1.0
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def mask_to_segments(mask: np.ndarray, epsilon_frac: float) -> list[list[float]]:
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
    path.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")


def load_rois(path: Path | None) -> dict[str, list[tuple[float, float]]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    rois: dict[str, list[tuple[float, float]]] = {}
    for stem, points in raw.items():
        rois[stem] = [(float(x), float(y)) for x, y in points]
    return rois


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]] | None) -> bool:
    if not polygon or len(polygon) < 3:
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = ((xj - xi) * (y - yi)) / max(1e-6, yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def main() -> None:
    args = parse_args()
    out = args.output_dir
    image_root = out / "images"
    label_root = out / "labels"
    for split in ["train", "val"]:
        (image_root / split).mkdir(parents=True, exist_ok=True)
        (label_root / split).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.teacher)
    rois = load_rois(args.roi_json)
    split_images = {"train": [], "val": []}
    exported_instances = 0
    exported_frames = 0
    rejected_by_roi = 0

    for video_idx, video_path in enumerate(args.videos):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        source_stem = video_path.stem
        roi = rois.get(source_stem)
        processed = 0
        frame_idx = 0
        while processed < args.max_frames_per_video:
            ok, original = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            frame, _ = resize_max_side(original, args.max_side)
            h, w = frame.shape[:2]
            result = model.predict(
                frame[:, :, ::-1],
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[0],
                device=args.device,
                verbose=False,
            )[0]
            label_lines: list[str] = []
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                frame_area = float(h * w)
                items = []
                for mask_small, conf, box in zip(masks, confs, boxes):
                    x1, y1, x2, y2 = box.tolist()
                    foot = ((x1 + x2) / (2 * max(1, w)), y2 / max(1, h))
                    if not point_in_polygon(foot, roi):
                        rejected_by_roi += 1
                        continue
                    mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    area_frac = float(mask.sum()) / max(1.0, frame_area)
                    if area_frac < args.min_area_frac or area_frac > args.max_area_frac:
                        continue
                    items.append((float(conf), mask))
                items.sort(reverse=True, key=lambda item: item[0])
                for _, mask in items[: args.max_objects]:
                    for segment in mask_to_segments(mask, args.contour_epsilon_frac):
                        label_lines.append("0 " + " ".join(f"{value:.6f}" for value in segment))
            if label_lines:
                split = "val" if processed % args.val_every == 0 else "train"
                stem = f"{video_idx:02d}_{source_stem}_{frame_idx:05d}"
                image_path = image_root / split / f"{stem}.jpg"
                label_path = label_root / split / f"{stem}.txt"
                cv2.imwrite(str(image_path), frame)
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                split_images[split].append(image_path)
                exported_frames += 1
                exported_instances += len(label_lines)
            processed += 1
            frame_idx += 1
        cap.release()

    write_split_list(split_images["train"], out / "train.txt")
    write_split_list(split_images["val"], out / "val.txt")
    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out.resolve().as_posix()}",
                "train: train.txt",
                "val: val.txt",
                "names:",
                "  0: player",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "teacher": args.teacher,
        "videos": [str(path) for path in args.videos],
        "train_frames": len(split_images["train"]),
        "val_frames": len(split_images["val"]),
        "instances": exported_instances,
        "frames": exported_frames,
        "rejected_by_roi": rejected_by_roi,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
