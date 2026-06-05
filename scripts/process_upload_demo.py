from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    misses: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-safe upload preview for the deployable demo site."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-tracks", type=int, default=12)
    parser.add_argument("--min-area-frac", type=float, default=0.002)
    return parser.parse_args()


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def detect_candidates(frame: np.ndarray, subtractor: cv2.BackgroundSubtractor, min_area: int) -> list[tuple[int, int, int, int]]:
    mask = subtractor.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    h, w = frame.shape[:2]
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area:
            continue
        if bw > 0.5 * w or bh > 0.8 * h:
            continue
        aspect = bw / max(1, bh)
        if aspect < 0.12 or aspect > 2.2:
            continue
        boxes.append((x, y, x + bw, y + bh))
    boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return boxes


def associate(tracks: list[Track], boxes: list[tuple[int, int, int, int]], next_id: int) -> tuple[list[Track], int]:
    assigned: set[int] = set()
    output: list[Track] = []
    for track in tracks:
        best_idx = -1
        best_iou = 0.0
        for idx, box in enumerate(boxes):
            if idx in assigned:
                continue
            iou = box_iou(track.box, box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= 0.08:
            assigned.add(best_idx)
            output.append(Track(track.track_id, boxes[best_idx], 0))
        elif track.misses < 5:
            output.append(Track(track.track_id, track.box, track.misses + 1))
    for idx, box in enumerate(boxes):
        if idx not in assigned:
            output.append(Track(next_id, box, 0))
            next_id += 1
    return output, next_id


def draw(frame: np.ndarray, tracks: list[Track], frame_idx: int, fps_estimate: float) -> np.ndarray:
    out = frame.copy()
    colors = [(89, 195, 106), (95, 183, 216), (226, 185, 93), (209, 113, 113), (177, 141, 231)]
    overlay = out.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.box
        color = colors[track.track_id % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(
            out,
            f"id={track.track_id}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    out = cv2.addWeighted(overlay, 0.16, out, 0.84, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (10, 12, 11), -1)
    cv2.putText(
        out,
        f"CPU upload preview | frame {frame_idx:03d} | tracks {len(tracks)} | {fps_estimate:.1f} FPS",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (244, 246, 241),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    frame = resize_max_side(frame, args.max_side)
    h, w = frame.shape[:2]
    min_area = int(h * w * args.min_area_frac)
    video_path = args.output_dir / "upload_preview.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), min(source_fps, 30.0), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {video_path}")

    subtractor = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=36, detectShadows=False)
    tracks: list[Track] = []
    next_id = 0
    frame_idx = 0
    timings: list[float] = []
    max_tracks_seen = 0
    active_frames = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        started = time.perf_counter()
        frame = resize_max_side(frame, args.max_side)
        boxes = detect_candidates(frame, subtractor, min_area)[: args.max_tracks]
        tracks, next_id = associate(tracks, boxes, next_id)
        tracks = tracks[: args.max_tracks]
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        fps_estimate = 1.0 / (sum(timings[-30:]) / len(timings[-30:]))
        writer.write(draw(frame, tracks, frame_idx, fps_estimate))
        max_tracks_seen = max(max_tracks_seen, len(tracks))
        active_frames += int(bool(tracks))
        frame_idx += 1

    cap.release()
    writer.release()
    avg = sum(timings) / len(timings) if timings else 0.0
    metrics = {
        "mode": "cpu_upload_preview",
        "video": str(args.video),
        "output_video": str(video_path),
        "frames": frame_idx,
        "source_fps": source_fps,
        "processing_fps": 1.0 / avg if avg else 0.0,
        "active_frame_pct": (active_frames / frame_idx * 100.0) if frame_idx else 0.0,
        "max_tracks_seen": max_tracks_seen,
        "total_track_ids": next_id,
        "note": "CPU-safe public demo preview. Full local GPU pipeline uses YOLO/LocateAnything plus EdgeTAM/SAM.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
