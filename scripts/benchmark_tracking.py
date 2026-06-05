from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the realtime tracking/SAM loop with rendering and debug I/O disabled."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/raw/youtube/clips/rec_league_0008_45s.mp4"),
    )
    parser.add_argument(
        "--boxes-json",
        type=Path,
        default=Path("examples/rec_league_0008_seed_boxes.json"),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("models/efficient_sam3_efficientvit_s_point_prompt_slim.pt"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max-sides", type=int, nargs="+", default=[384, 512, 640])
    parser.add_argument("--sam-every", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--detector", choices=["seed", "yolo"], default="seed")
    parser.add_argument("--yolo-every", type=int, default=10)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--sam-on-yolo", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[idx]


def summarize_metrics(metrics_path: Path, target_fps: float, warmup_frames: int) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_frames = metrics["frames"]
    frames = [frame for frame in all_frames if int(frame["frame"]) >= warmup_frames]
    frame_times = [float(frame["frame_seconds"]) for frame in frames]
    sam_times = [
        float(frame["sam_seconds"])
        for frame in frames
        if float(frame.get("sam_seconds", 0.0)) > 0
    ]
    track_times = [float(frame.get("track_seconds", 0.0)) for frame in frames]
    active_counts = [len(frame.get("boxes", [])) for frame in frames]
    active_frames = sum(1 for count in active_counts if count > 0)
    processed = len(frame_times)
    total = sum(frame_times)
    fps = processed / total if total > 0 else 0.0
    target_frame_ms = 1000.0 / target_fps
    return {
        "frames": processed,
        "warmup_frames_excluded": warmup_frames,
        "fps": fps,
        "target_fps": target_fps,
        "meets_target": fps >= target_fps and percentile(frame_times, 95) * 1000 <= target_frame_ms,
        "frame_ms_avg": statistics.fmean(frame_times) * 1000 if frame_times else 0.0,
        "frame_ms_p50": percentile(frame_times, 50) * 1000,
        "frame_ms_p95": percentile(frame_times, 95) * 1000,
        "track_ms_avg": statistics.fmean(track_times) * 1000 if track_times else 0.0,
        "active_frame_pct": (active_frames / processed * 100.0) if processed else 0.0,
        "avg_tracks": statistics.fmean(active_counts) if active_counts else 0.0,
        "sam_ms_avg": statistics.fmean(sam_times) * 1000 if sam_times else 0.0,
        "sam_ms_p95": percentile(sam_times, 95) * 1000,
        "sam_calls": len(sam_times),
    }


def run_case(args: argparse.Namespace, max_side: int, sam_every: int) -> dict[str, Any]:
    case_name = f"side{max_side}_sam{sam_every}"
    output_dir = args.output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/track_segment_video.py",
        "--video",
        str(args.video),
        "--sam-checkpoint",
        str(args.sam_checkpoint),
        "--output-dir",
        str(output_dir),
        "--max-frames",
        str(args.frames),
        "--max-side",
        str(max_side),
        "--sam-every",
        str(sam_every),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--no-write-frames",
        "--no-render-overlays",
        "--no-print-metrics",
        "--allow-empty-tracks",
    ]
    if args.detector == "seed":
        cmd.extend(["--boxes-json", str(args.boxes_json)])
    else:
        cmd.extend(
            [
                "--yolo-every",
                str(args.yolo_every),
                "--yolo-model",
                args.yolo_model,
                "--yolo-conf",
                str(args.yolo_conf),
                "--yolo-imgsz",
                str(args.yolo_imgsz),
            ]
        )
        if not args.sam_on_yolo:
            cmd.append("--no-sam-on-yolo")
    subprocess.run(cmd, check=True)
    summary = summarize_metrics(output_dir / "metrics.json", args.target_fps, args.warmup_frames)
    summary.update(
        {
            "case": case_name,
            "detector": args.detector,
            "max_side": max_side,
            "sam_every": sam_every,
            "yolo_every": args.yolo_every if args.detector == "yolo" else None,
            "sam_on_yolo": args.sam_on_yolo if args.detector == "yolo" else None,
            "output_dir": str(output_dir),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for max_side in args.max_sides:
        for sam_every in args.sam_every:
            results.append(run_case(args, max_side, sam_every))

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(
        "case,max_side,sam_every,fps,frame_ms_p50,frame_ms_p95,"
        "sam_ms_avg,sam_ms_p95,sam_calls,active_frame_pct,avg_tracks,meets_target"
    )
    for row in results:
        print(
            f"{row['case']},{row['max_side']},{row['sam_every']},"
            f"{row['fps']:.2f},{row['frame_ms_p50']:.2f},{row['frame_ms_p95']:.2f},"
            f"{row['sam_ms_avg']:.2f},{row['sam_ms_p95']:.2f},"
            f"{row['sam_calls']},{row['active_frame_pct']:.1f},"
            f"{row['avg_tracks']:.2f},{row['meets_target']}"
        )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
