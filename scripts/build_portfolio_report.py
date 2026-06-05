from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portfolio-ready project report from saved benchmark outputs."
    )
    parser.add_argument("--output-md", type=Path, default=Path("PORTFOLIO.md"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/portfolio_report/summary.json"))
    parser.add_argument(
        "--sav-comparison",
        type=Path,
        default=Path("outputs/sav_comparison_3sample/comparison.json"),
    )
    parser.add_argument(
        "--sports-yolo",
        type=Path,
        default=Path("outputs/edgetam_yolo26n_rec_league_seed_compare/metrics.json"),
    )
    parser.add_argument(
        "--sports-locate",
        type=Path,
        default=Path("outputs/edgetam_locate_rec_league_seed_compare/metrics.json"),
    )
    parser.add_argument(
        "--realtime-summary",
        type=Path,
        default=Path("outputs/benchmarks_realtime_yolo480_identity_missbuf/side512_sam30/metrics.json"),
    )
    parser.add_argument(
        "--analytics-summary",
        type=Path,
        default=Path("outputs/benchmarks_realtime_yolo480_identity_missbuf/side512_sam30/analytics_summary.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_realtime(metrics: dict[str, Any]) -> dict[str, float]:
    frames = metrics.get("frames", [])
    frame_seconds = [float(frame.get("frame_seconds", 0.0)) for frame in frames[10:]]
    active = [frame for frame in frames[10:] if frame.get("boxes")]
    return {
        "frames": float(len(frames)),
        "fps": 1.0 / (sum(frame_seconds) / len(frame_seconds)) if frame_seconds else 0.0,
        "p50_ms": percentile(frame_seconds, 50) * 1000.0,
        "p95_ms": percentile(frame_seconds, 95) * 1000.0,
        "active_frame_pct": (len(active) / len(frames[10:]) * 100.0) if len(frames) > 10 else 0.0,
        "avg_tracks": (
            sum(len(frame.get("boxes", [])) for frame in frames[10:]) / len(frames[10:])
            if len(frames) > 10
            else 0.0
        ),
    }


def summarize_seed(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": metrics.get("seed_detector", "unknown"),
        "seed_seconds": float(metrics.get("seed_seconds", 0.0)),
        "raw_boxes": int(metrics.get("seed_filter_stats", {}).get("raw", 0)),
        "filtered_boxes": int(metrics.get("seed_filter_stats", {}).get("filtered", 0)),
        "kept_boxes": int(metrics.get("seed_filter_stats", {}).get("kept", len(metrics.get("seed_boxes", [])))),
        "propagate_fps": float(metrics.get("propagate_fps", 0.0)),
        "end_to_end_seconds": float(metrics.get("seed_seconds", 0.0))
        + float(metrics.get("build_seconds", 0.0))
        + float(metrics.get("init_seconds", 0.0))
        + float(metrics.get("propagate_seconds", 0.0)),
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sav_rows = read_json(args.sav_comparison)
    yolo_seed = summarize_seed(read_json(args.sports_yolo))
    locate_seed = summarize_seed(read_json(args.sports_locate))
    realtime = summarize_realtime(read_json(args.realtime_summary))
    analytics = read_json(args.analytics_summary)

    long_tracks = sum(1 for track in analytics.get("tracks", {}).values() if int(track.get("frames", 0)) >= 60)
    report = {
        "project": "Fast Sports Segmentation",
        "positioning": "Realtime sports-video segmentation and analytics using YOLO/LocateAnything seeding, EdgeTAM mask propagation, and court-space analytics.",
        "sports_realtime": realtime,
        "seed_comparison": [yolo_seed, locate_seed],
        "sav_comparison": sav_rows,
        "analytics": {
            "frames": analytics.get("frames"),
            "rows": analytics.get("rows"),
            "track_count": analytics.get("track_count"),
            "long_tracks_60_plus": long_tracks,
            "court_units": analytics.get("homography", {}).get("units"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    seed_table = markdown_table(
        [
            "Seed Detector",
            "Seed Time",
            "Raw Boxes",
            "Filtered",
            "Kept",
            "EdgeTAM FPS",
            "End-to-End",
        ],
        [
            [
                row["detector"],
                f"{row['seed_seconds']:.2f}s",
                str(row["raw_boxes"]),
                str(row["filtered_boxes"]),
                str(row["kept_boxes"]),
                f"{row['propagate_fps']:.2f}",
                f"{row['end_to_end_seconds']:.2f}s",
            ]
            for row in [yolo_seed, locate_seed]
        ],
    )
    sav_table = markdown_table(
        [
            "System",
            "Cadence",
            "Mask Pairs",
            "Mean Mask IoU",
            "Mask J@0.5",
            "Mean Box IoU",
            "FPS",
        ],
        [
            [
                row["name"],
                row["cadence"],
                str(row["mask_pairs"]),
                f"{row['mean_mask_iou']:.3f}",
                f"{row['mask_j_at_50']:.3f}",
                f"{row['mean_box_iou']:.3f}",
                f"{row['fps']:.2f}",
            ]
            for row in sav_rows
        ],
    )

    md = f"""# Fast Sports Segmentation Portfolio Report

## Resume Positioning

Built a realtime sports-video segmentation and analytics pipeline using YOLO/LocateAnything seeding, EdgeTAM video mask propagation, and court-space tracking analytics; benchmarked against EfficientSAM3 on SA-V with mask IoU, J@0.5, and FPS tradeoff analysis.

## What The System Does

- Detects player seed boxes with YOLO by default, with LocateAnything available for slower open-vocabulary recovery.
- Propagates masks through video with EdgeTAM.
- Exports per-frame tracking geometry and approximate court-space analytics.
- Benchmarks segmentation quality on SA-V and speed on local sports clips.

## Sports Realtime Profile

Measured on `rec_league_0008_45s.mp4` with YOLOv8n, persistent IDs, BF16, 512-side frames, YOLO every 10 frames, and SAM refresh every 30 frames:

- FPS after warmup: `{realtime['fps']:.2f}`
- p50 frame time: `{realtime['p50_ms']:.2f} ms`
- p95 frame time: `{realtime['p95_ms']:.2f} ms`
- Active-frame coverage: `{realtime['active_frame_pct']:.1f}%`
- Average tracks: `{realtime['avg_tracks']:.2f}`

Analytics export:

- Frames: `{analytics.get('frames')}`
- Analytics rows: `{analytics.get('rows')}`
- Track IDs: `{analytics.get('track_count')}`
- Tracks lasting at least 60 frames: `{long_tracks}`
- Court projection units: `{analytics.get('homography', {}).get('units')}`

## Seed Detector Comparison

Measured on the same 60-frame rec-league EdgeTAM run:

{seed_table}

YOLO is the correct default for realtime player seeding. LocateAnything is useful as an async semantic recovery path, not as a live seed detector.

## SA-V Accuracy Comparison

All rows use the same 3-video SA-V Subset 51 shard with oracle first-frame boxes.

{sav_table}

Interpretation: EdgeTAM is the accuracy winner by a wide margin. EfficientSAM3 is faster in the sparse template-tracking setup, but without video memory it is not competitive on mask tracking quality.

## Portfolio Claims Supported By Metrics

- Built and benchmarked an end-to-end video segmentation system rather than only running a model.
- Chose EdgeTAM over EfficientSAM3 for mask propagation based on measured SA-V accuracy.
- Quantified detector tradeoffs: YOLO seed latency is seconds, LocateAnything seed latency is tens of seconds on this host.
- Added court-space analytics and persistent IDs to connect segmentation output to sports analytics.

## Next Work

- Add ByteTrack/DeepSORT-grade identity association for crowded sports footage.
- Add periodic EdgeTAM correction prompts from YOLO when players enter or occlude.
- Expand SA-V evaluation from 3 videos to 25-50 videos with confidence intervals.
- Export a polished 15-30 second demo clip with side-by-side masks, IDs, and court analytics.
"""
    args.output_md.write_text(md, encoding="utf-8")
    print(f"wrote {args.output_md} and {args.output_json}")


if __name__ == "__main__":
    main()
