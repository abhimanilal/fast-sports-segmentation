from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export image-space sports analytics from tracking metrics."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def box_features(box: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
    }


def main() -> None:
    args = parse_args()
    metrics: dict[str, Any] = json.loads(args.metrics.read_text(encoding="utf-8"))
    fps = args.fps or float(metrics.get("source_fps") or 30.0)
    last_by_track: dict[int, tuple[int, float, float]] = {}
    rows: list[dict[str, Any]] = []
    track_frame_counts: defaultdict[int, int] = defaultdict(int)
    speeds_by_track: defaultdict[int, list[float]] = defaultdict(list)

    for frame in metrics["frames"]:
        frame_idx = int(frame["frame"])
        boxes = frame.get("boxes", [])
        track_scores = frame.get("track_scores", [])
        sam_scores = frame.get("sam_scores", [])
        detection_source = frame.get("detection_source") or ""
        if detection_source:
            # Current track ids are local to the active detector seed set. Until a
            # real association layer exists, do not carry velocity across reseeds.
            last_by_track.clear()
        for local_track_id, box in enumerate(boxes):
            features = box_features([float(value) for value in box])
            prev = last_by_track.get(local_track_id)
            speed_px_s = 0.0
            if prev is not None:
                prev_frame, prev_cx, prev_cy = prev
                dt = max(1, frame_idx - prev_frame) / fps
                speed_px_s = math.hypot(features["cx"] - prev_cx, features["cy"] - prev_cy) / dt
                speeds_by_track[local_track_id].append(speed_px_s)
            last_by_track[local_track_id] = (frame_idx, features["cx"], features["cy"])
            track_frame_counts[local_track_id] += 1
            rows.append(
                {
                    "frame": frame_idx,
                    "time_seconds": frame_idx / fps,
                    "track_id": local_track_id,
                    **features,
                    "track_score": track_scores[local_track_id]
                    if local_track_id < len(track_scores)
                    else "",
                    "sam_score": sam_scores[local_track_id]
                    if local_track_id < len(sam_scores)
                    else "",
                    "speed_px_s": speed_px_s,
                    "detection_source": detection_source,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "time_seconds",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "cx",
                "cy",
                "width",
                "height",
                "area",
                "track_score",
                "sam_score",
                "speed_px_s",
                "detection_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "metrics": str(args.metrics),
        "fps": fps,
        "frames": len(metrics["frames"]),
        "rows": len(rows),
        "active_frame_pct": (
            sum(1 for frame in metrics["frames"] if frame.get("boxes"))
            / max(1, len(metrics["frames"]))
            * 100.0
        ),
        "track_count": len(track_frame_counts),
        "tracks": {
            str(track_id): {
                "frames": count,
                "mean_speed_px_s": (
                    sum(speeds_by_track[track_id]) / len(speeds_by_track[track_id])
                    if speeds_by_track[track_id]
                    else 0.0
                ),
                "max_speed_px_s": max(speeds_by_track[track_id])
                if speeds_by_track[track_id]
                else 0.0,
            }
            for track_id, count in sorted(track_frame_counts.items())
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
