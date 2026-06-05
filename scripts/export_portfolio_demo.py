from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a polished side-by-side portfolio demo clip from saved sports segmentation artifacts."
    )
    parser.add_argument("--raw-video", type=Path, default=Path("data/raw/youtube/clips/rec_league_0008_45s.mp4"))
    parser.add_argument(
        "--mask-video",
        type=Path,
        default=Path("outputs/edgetam_yolo26n_rec_league_seed_compare/tracked.mp4"),
    )
    parser.add_argument(
        "--edge-metrics",
        type=Path,
        default=Path("outputs/edgetam_yolo26n_rec_league_seed_compare/metrics.json"),
    )
    parser.add_argument(
        "--sav-comparison",
        type=Path,
        default=Path("outputs/sav_comparison_3sample/comparison.json"),
    )
    parser.add_argument(
        "--analytics-csv",
        type=Path,
        default=Path("outputs/benchmarks_realtime_yolo480_identity_missbuf/side512_sam30/analytics.csv"),
    )
    parser.add_argument(
        "--portfolio-summary",
        type=Path,
        default=Path("outputs/portfolio_report/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/portfolio_demo"))
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--output-height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=29.97)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))))


def resize_to_height(frame: np.ndarray, output_height: int) -> np.ndarray:
    if output_height <= 0:
        return frame
    height, width = frame.shape[:2]
    if height == output_height:
        return frame
    scale = output_height / float(height)
    output_width = int(round(width * scale))
    if output_width % 2:
        output_width += 1
    if output_height % 2:
        output_height += 1
    return cv2.resize(frame, (output_width, output_height))


def read_analytics(csv_path: Path) -> dict[int, dict[str, float]]:
    by_frame: dict[int, dict[str, float]] = {}
    if not csv_path.exists():
        return by_frame
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_by_frame: dict[int, list[dict[str, str]]] = {}
        for row in reader:
            rows_by_frame.setdefault(int(row["frame"]), []).append(row)
    for frame, rows in rows_by_frame.items():
        speeds = [float(row["speed_px_s"]) for row in rows if row.get("speed_px_s")]
        court_speeds = [
            float(row["court_speed_units_s"])
            for row in rows
            if row.get("court_speed_units_s") not in ("", None)
        ]
        by_frame[frame] = {
            "tracks": float(len({row["track_id"] for row in rows})),
            "mean_speed_px_s": float(np.mean(speeds)) if speeds else 0.0,
            "mean_court_speed": float(np.mean(court_speeds)) if court_speeds else 0.0,
        }
    return by_frame


def draw_panel(
    width: int,
    height: int,
    frame_idx: int,
    edge_metrics: dict[str, Any],
    sav_rows: list[dict[str, Any]],
    portfolio: dict[str, Any],
    analytics_by_frame: dict[int, dict[str, float]],
) -> np.ndarray:
    panel = np.full((height, width, 3), (22, 24, 27), dtype=np.uint8)
    accent = (72, 187, 120)
    secondary = (185, 194, 205)
    white = (245, 247, 250)
    muted = (125, 137, 151)

    def text(line: str, x: int, y: int, scale: float = 0.46, color=white, thickness: int = 1) -> int:
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return y + int(24 * scale / 0.46)

    y = 34
    y = text("FAST SPORTS SEGMENTATION", 18, y, 0.55, accent, 2)
    y = text("YOLO seed -> EdgeTAM masks -> court analytics", 18, y + 4, 0.42, secondary)
    cv2.line(panel, (18, y + 8), (width - 18, y + 8), (58, 64, 72), 1)
    y += 40

    realtime = portfolio.get("sports_realtime", {})
    analytics = analytics_by_frame.get(frame_idx, {})
    stats = [
        ("Frame", f"{frame_idx:03d}"),
        ("Visible tracks", f"{int(analytics.get('tracks', 0))}"),
        ("Realtime loop", f"{realtime.get('fps', 0):.1f} FPS"),
        ("p95 frame", f"{realtime.get('p95_ms', 0):.1f} ms"),
        ("Seed time", f"{edge_metrics.get('seed_seconds', 0):.2f}s"),
        ("EdgeTAM prop.", f"{edge_metrics.get('propagate_fps', 0):.2f} FPS"),
    ]
    for label, value in stats:
        y = text(label.upper(), 18, y, 0.34, muted)
        y = text(value, 18, y + 1, 0.58, white, 2)
        y += 6

    cv2.line(panel, (18, y + 5), (width - 18, y + 5), (58, 64, 72), 1)
    y += 34
    best = next((row for row in sav_rows if row["name"].startswith("EdgeTAM")), {})
    y = text("SA-V VALIDATION", 18, y, 0.42, accent, 1)
    y = text(f"Mask IoU: {best.get('mean_mask_iou', 0):.3f}", 18, y + 4, 0.48, white, 1)
    y = text(f"J@0.5: {best.get('mask_j_at_50', 0):.3f}", 18, y, 0.48, white, 1)
    y = text("EdgeTAM selected over EfficientSAM3 for quality", 18, y + 8, 0.34, secondary, 1)

    cv2.rectangle(panel, (14, 14), (width - 14, height - 14), (58, 64, 72), 1)
    return panel


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 247, 250), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_metrics = read_json(args.edge_metrics)
    sav_rows = read_json(args.sav_comparison)
    portfolio = read_json(args.portfolio_summary)
    analytics_by_frame = read_analytics(args.analytics_csv)

    raw_cap = cv2.VideoCapture(str(args.raw_video))
    mask_cap = cv2.VideoCapture(str(args.mask_video))
    if not raw_cap.isOpened():
        raise RuntimeError(f"Could not open raw video {args.raw_video}")
    if not mask_cap.isOpened():
        raise RuntimeError(f"Could not open mask video {args.mask_video}")

    ok_raw, raw = raw_cap.read()
    ok_mask, mask = mask_cap.read()
    if not ok_raw or not ok_mask:
        raise RuntimeError("Could not read first demo frames")
    raw = resize_max_side(raw, args.max_side)
    mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]))
    h, w = raw.shape[:2]
    raw = resize_to_height(raw, args.output_height)
    mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]))
    h, w = raw.shape[:2]
    panel_w = 420
    canvas_size = (w * 2 + panel_w, h)
    if canvas_size[0] % 2 or canvas_size[1] % 2:
        canvas_size = (canvas_size[0] + canvas_size[0] % 2, canvas_size[1] + canvas_size[1] % 2)
    output_video = args.output_dir / "sports_segmentation_demo.mp4"
    if output_video.exists():
        output_video.unlink()
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        canvas_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video} at {canvas_size}")

    frame_idx = 0
    raw_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    mask_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    snapshots = {0, max(0, args.max_frames // 2), max(0, args.max_frames - 1)}
    while frame_idx < args.max_frames:
        ok_raw, raw = raw_cap.read()
        ok_mask, mask = mask_cap.read()
        if not ok_raw or not ok_mask:
            break
        raw = resize_max_side(raw, args.max_side)
        raw = resize_to_height(raw, args.output_height)
        mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]))
        left = label_frame(raw, "RAW CLIP")
        middle = label_frame(mask, "EDGETAM MASK PROPAGATION")
        panel = draw_panel(
            panel_w,
            raw.shape[0],
            frame_idx,
            edge_metrics,
            sav_rows,
            portfolio,
            analytics_by_frame,
        )
        canvas = np.hstack([left, middle, panel])
        if canvas.shape[1] != canvas_size[0] or canvas.shape[0] != canvas_size[1]:
            canvas = cv2.resize(canvas, canvas_size)
        writer.write(canvas)
        if frame_idx in snapshots:
            cv2.imwrite(str(args.output_dir / f"demo_frame_{frame_idx:03d}.jpg"), canvas)
        frame_idx += 1

    raw_cap.release()
    mask_cap.release()
    writer.release()
    summary = {
        "output_video": str(output_video),
        "frames": frame_idx,
        "fps": args.fps,
        "canvas_width": canvas_size[0],
        "canvas_height": canvas_size[1],
        "snapshots": sorted(snapshots),
        "source_raw": str(args.raw_video),
        "source_mask": str(args.mask_video),
    }
    (args.output_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary['output_video']} with {frame_idx} frames")


if __name__ == "__main__":
    main()
