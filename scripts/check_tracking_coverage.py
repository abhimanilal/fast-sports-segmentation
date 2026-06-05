from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail a tracking run when boxes or rendered masks visibly collapse."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--min-avg-boxes", type=float, default=5.0)
    parser.add_argument("--min-avg-masks", type=float, default=5.0)
    parser.add_argument("--max-maskless-frames", type=int, default=0)
    parser.add_argument("--max-low-mask-frames", type=int, default=30)
    parser.add_argument("--low-mask-threshold", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    frames = metrics.get("frames", [])
    if not frames:
        raise SystemExit(f"No frame metrics found in {args.metrics}")

    box_counts = [len(frame.get("boxes", [])) for frame in frames]
    mask_counts = [int(frame.get("rendered_mask_count", 0)) for frame in frames]
    summary = {
        "frames": len(frames),
        "avg_boxes": round(statistics.mean(box_counts), 3),
        "min_boxes": min(box_counts),
        "avg_rendered_masks": round(statistics.mean(mask_counts), 3),
        "min_rendered_masks": min(mask_counts),
        "maskless_frames": sum(count == 0 for count in mask_counts),
        "low_mask_frames": sum(count < args.low_mask_threshold for count in mask_counts),
        "unique_track_ids": len({track_id for frame in frames for track_id in frame.get("track_ids", [])}),
    }
    print(json.dumps(summary, indent=2))

    failures: list[str] = []
    if summary["avg_boxes"] < args.min_avg_boxes:
        failures.append(f"avg boxes {summary['avg_boxes']} < {args.min_avg_boxes}")
    if summary["avg_rendered_masks"] < args.min_avg_masks:
        failures.append(f"avg masks {summary['avg_rendered_masks']} < {args.min_avg_masks}")
    if summary["maskless_frames"] > args.max_maskless_frames:
        failures.append(f"maskless frames {summary['maskless_frames']} > {args.max_maskless_frames}")
    if summary["low_mask_frames"] > args.max_low_mask_frames:
        failures.append(f"low-mask frames {summary['low_mask_frames']} > {args.max_low_mask_frames}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
