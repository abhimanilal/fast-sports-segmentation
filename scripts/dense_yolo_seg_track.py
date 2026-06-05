from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


@dataclass
class Detection:
    box: np.ndarray
    mask: np.ndarray
    conf: float
    hist: np.ndarray
    foot: tuple[float, float]


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    mask: np.ndarray
    hist: np.ndarray
    conf: float
    age: int = 0
    misses: int = 0
    hits: int = 1
    velocity: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense player segmentation tracking with YOLO masks and custom temporal association."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=450)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-detections", type=int, default=12)
    parser.add_argument("--max-track-misses", type=int, default=12)
    parser.add_argument("--min-area-frac", type=float, default=0.0012)
    parser.add_argument("--max-area-frac", type=float, default=0.32)
    parser.add_argument("--association-threshold", type=float, default=0.46)
    parser.add_argument(
        "--roi-polygon",
        default="",
        help='Optional active-play polygon as "x1,y1 x2,y2 ..."; detections are kept when the box foot point is inside.',
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=1.0,
        help="Scale ROI coordinates before applying them. Useful when a polygon was measured at another resize.",
    )
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-metrics", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_roi_polygon(raw: str, scale: float) -> np.ndarray | None:
    if not raw.strip():
        return None
    points: list[list[float]] = []
    for token in raw.replace(";", " ").split():
        x_raw, y_raw = token.split(",", maxsplit=1)
        points.append([float(x_raw) * scale, float(y_raw) * scale])
    if len(points) < 3:
        raise ValueError("--roi-polygon needs at least three x,y points")
    return np.asarray(points, dtype=np.float32)


def foot_point(box: np.ndarray) -> tuple[float, float]:
    return (float((box[0] + box[2]) / 2.0), float(box[3]))


def inside_roi(point: tuple[float, float], roi_polygon: np.ndarray | None) -> bool:
    if roi_polygon is None:
        return True
    return cv2.pointPolygonTest(roi_polygon, point, measureDist=False) >= 0


def resize_max_side(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return frame, 1.0
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def center_distance_frac(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    ac = np.asarray([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0], dtype=np.float32)
    bc = np.asarray([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)
    return float(np.linalg.norm(ac - bc) / max(1.0, np.hypot(width, height)))


def appearance_hist(frame: np.ndarray, box: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    x1, y1, x2, y2 = np.clip(box.astype(int), [0, 0, 1, 1], [frame.shape[1] - 1, frame.shape[0] - 1, frame.shape[1], frame.shape[0]])
    if x2 <= x1 or y2 <= y1:
        return np.zeros(48, dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop_mask = None
    if mask is not None:
        crop_mask = mask[y1:y2, x1:x2].astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1], crop_mask, [16, 3], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm > 0 else hist


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return 0.0
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def decode_detections(
    result,
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
    roi_polygon: np.ndarray | None,
) -> tuple[list[Detection], dict[str, int]]:
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    detections: list[Detection] = []
    stats = {"raw": 0, "class": 0, "shape": 0, "roi": 0, "kept": 0}
    if result.boxes is None or result.masks is None:
        return detections, stats
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    for box, conf, cls, mask_small in zip(boxes, confs, classes, masks):
        stats["raw"] += 1
        if int(cls) != 0:
            continue
        stats["class"] += 1
        box = box.astype(np.float32)
        area_frac = box_area(box) / frame_area
        if area_frac < args.min_area_frac or area_frac > args.max_area_frac:
            stats["shape"] += 1
            continue
        foot = foot_point(box)
        if not inside_roi(foot, roi_polygon):
            stats["roi"] += 1
            continue
        mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        detections.append(Detection(box=box, mask=mask, conf=float(conf), hist=appearance_hist(frame_bgr, box, mask), foot=foot))
    detections.sort(key=lambda det: det.conf * box_area(det.box), reverse=True)
    kept = detections[: args.max_detections]
    stats["kept"] = len(kept)
    return kept, stats


def predict_box(track: Track, width: int, height: int) -> np.ndarray:
    if track.velocity is None:
        return track.box
    predicted = track.box + np.asarray([track.velocity[0], track.velocity[1], track.velocity[0], track.velocity[1]], dtype=np.float32)
    predicted[[0, 2]] = np.clip(predicted[[0, 2]], 0, width - 1)
    predicted[[1, 3]] = np.clip(predicted[[1, 3]], 0, height - 1)
    return predicted


def association_score(track: Track, detection: Detection, width: int, height: int) -> float:
    predicted = predict_box(track, width, height)
    iou = box_iou(predicted, detection.box)
    center = 1.0 - min(1.0, center_distance_frac(predicted, detection.box, width, height) / 0.28)
    appearance = hist_similarity(track.hist, detection.hist)
    return 0.55 * iou + 0.30 * center + 0.15 * appearance


def associate_tracks(
    frame_bgr: np.ndarray,
    tracks: list[Track],
    detections: list[Detection],
    next_track_id: int,
    args: argparse.Namespace,
) -> tuple[list[Track], int, dict[str, int]]:
    h, w = frame_bgr.shape[:2]
    stats = {"detections": len(detections), "matched": 0, "new": 0, "kept_missed": 0, "dropped": 0}
    if not tracks:
        created = [
            Track(track_id=next_track_id + idx, box=det.box, mask=det.mask, hist=det.hist, conf=det.conf)
            for idx, det in enumerate(detections)
        ]
        stats["new"] = len(created)
        return created, next_track_id + len(created), stats

    score_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for track_idx, track in enumerate(tracks):
        for det_idx, detection in enumerate(detections):
            score_matrix[track_idx, det_idx] = association_score(track, detection, w, h)

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    output: list[Track] = []
    if detections:
        rows, cols = linear_sum_assignment(1.0 - score_matrix)
        for row, col in zip(rows.tolist(), cols.tolist()):
            score = float(score_matrix[row, col])
            if score < args.association_threshold:
                continue
            track = tracks[row]
            detection = detections[col]
            old_center = np.asarray([(track.box[0] + track.box[2]) / 2.0, (track.box[1] + track.box[3]) / 2.0], dtype=np.float32)
            new_center = np.asarray([(detection.box[0] + detection.box[2]) / 2.0, (detection.box[1] + detection.box[3]) / 2.0], dtype=np.float32)
            velocity = new_center - old_center
            if track.velocity is not None:
                velocity = 0.65 * track.velocity + 0.35 * velocity
            track.box = detection.box
            track.mask = detection.mask
            track.hist = 0.75 * track.hist + 0.25 * detection.hist
            norm = float(np.linalg.norm(track.hist))
            if norm > 0:
                track.hist = track.hist / norm
            track.conf = detection.conf
            track.velocity = velocity
            track.age += 1
            track.misses = 0
            track.hits += 1
            output.append(track)
            matched_tracks.add(row)
            matched_detections.add(col)
            stats["matched"] += 1

    for idx, track in enumerate(tracks):
        if idx in matched_tracks:
            continue
        track.misses += 1
        track.age += 1
        if track.misses <= args.max_track_misses:
            output.append(track)
            stats["kept_missed"] += 1
        else:
            stats["dropped"] += 1

    for idx, detection in enumerate(detections):
        if idx in matched_detections:
            continue
        output.append(Track(track_id=next_track_id, box=detection.box, mask=detection.mask, hist=detection.hist, conf=detection.conf))
        next_track_id += 1
        stats["new"] += 1

    output.sort(key=lambda track: (track.misses > 0, track.track_id))
    return output, next_track_id, stats


def overlay(frame_bgr: np.ndarray, tracks: list[Track]) -> np.ndarray:
    colors = np.asarray(
        [
            [82, 209, 111],
            [79, 178, 226],
            [231, 190, 78],
            [229, 109, 91],
            [172, 140, 232],
            [62, 204, 184],
        ],
        dtype=np.uint8,
    )
    out = frame_bgr.copy()
    for track in tracks:
        if track.misses > 2:
            continue
        color = colors[track.track_id % len(colors)]
        mask = track.mask.astype(bool)
        out[mask] = (0.56 * out[mask] + 0.44 * color).astype(np.uint8)
    for track in tracks:
        if track.misses > 2:
            continue
        color = colors[track.track_id % len(colors)].tolist()
        x1, y1, x2, y2 = track.box.astype(int).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"id={track.track_id}"
        if track.misses:
            label += f" m={track.misses}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_polygon = parse_roi_polygon(args.roi_polygon, args.roi_scale)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    model = YOLO(args.model)

    tracks: list[Track] = []
    next_track_id = 0
    writer = None
    metrics = {
        "mode": "dense_yolo_seg_track",
        "video": str(args.video),
        "model": args.model,
        "source_fps": source_fps,
        "roi_polygon": roi_polygon.astype(float).tolist() if roi_polygon is not None else None,
        "frames": [],
    }

    for local_idx in range(args.max_frames):
        ok, original = cap.read()
        if not ok:
            break
        started = time.perf_counter()
        frame_bgr, scale = resize_max_side(original, args.max_side)
        result = model.predict(
            frame_bgr[:, :, ::-1],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=[0],
            device=args.device,
            verbose=False,
        )[0]
        detections, detection_stats = decode_detections(result, frame_bgr, args, roi_polygon)
        tracks, next_track_id, assoc = associate_tracks(frame_bgr, tracks, detections, next_track_id, args)
        visible_tracks = [track for track in tracks if track.misses <= 2]
        vis = overlay(frame_bgr, visible_tracks)

        if args.write_video:
            if writer is None:
                h, w = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output_dir / "tracked.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (w, h),
                )
            writer.write(vis)
        if args.write_frames:
            cv2.imwrite(str(args.output_dir / f"frame_{args.start_frame + local_idx:05d}.jpg"), vis)

        metrics["frames"].append(
            {
                "frame": args.start_frame + local_idx,
                "detections": len(detections),
                "visible_tracks": len(visible_tracks),
                "track_ids": [track.track_id for track in visible_tracks],
                "track_misses": [track.misses for track in visible_tracks],
                "boxes": [track.box.astype(float).tolist() for track in visible_tracks],
                "mask_count": sum(int(track.mask.any()) for track in visible_tracks),
                "detection_stats": detection_stats,
                "association": assoc,
                "frame_seconds": time.perf_counter() - started,
                "scale": scale,
            }
        )

    cap.release()
    if writer is not None:
        writer.release()
        metrics["video_output"] = str(args.output_dir / "tracked.mp4")
    metrics["total_track_ids"] = next_track_id
    metrics["frames_processed"] = len(metrics["frames"])
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.print_metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
