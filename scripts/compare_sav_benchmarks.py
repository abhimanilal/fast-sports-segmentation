from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SA-V benchmark summary JSON files.")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def row_from_summary(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    system = raw.get("system", path.parent.name)
    if "edgetam" in system:
        return {
            "name": "EdgeTAM oracle boxes",
            "summary": str(path),
            "mask_pairs": raw.get("evaluated_pairs", 0),
            "box_pairs": raw.get("evaluated_pairs", 0),
            "mean_mask_iou": raw.get("mean_mask_iou", 0.0),
            "mean_box_iou": raw.get("mean_box_iou", 0.0),
            "mask_j_at_50": raw.get("j_at_50", 0.0),
            "box_j_at_50": raw.get("j_at_50", 0.0),
            "fps": raw.get("propagate_fps", 0.0),
            "cadence": "video memory every frame",
        }
    cadence = raw.get("sam_every")
    return {
        "name": f"EfficientSAM sparse refresh {cadence}",
        "summary": str(path),
        "mask_pairs": raw.get("mask_refresh_pairs", 0),
        "box_pairs": raw.get("box_pairs", 0),
        "mean_mask_iou": raw.get("mean_mask_iou_refresh", 0.0),
        "mean_box_iou": raw.get("mean_box_iou_all", 0.0),
        "mask_j_at_50": raw.get("mask_j_at_50_refresh", 0.0),
        "box_j_at_50": raw.get("box_j_at_50_all", 0.0),
        "fps": raw.get("fps", 0.0),
        "cadence": f"SAM every {cadence} frames",
    }


def main() -> None:
    args = parse_args()
    rows = [row_from_summary(path) for path in args.summary]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "| System | Cadence | Mask pairs | Box pairs | Mean mask IoU | Mean box IoU | Mask J@0.5 | Box J@0.5 | FPS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {cadence} | {mask_pairs} | {box_pairs} | {mean_mask_iou:.3f} | "
            "{mean_box_iou:.3f} | {mask_j_at_50:.3f} | {box_j_at_50:.3f} | {fps:.2f} |".format(
                **row
            )
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_md} and {args.output_json}")


if __name__ == "__main__":
    main()
