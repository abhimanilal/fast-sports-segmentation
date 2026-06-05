from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO detector seed-box accuracy on SA-V masklet boxes."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/sav_subset51_shard_3/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sav_yolo_detector_comparison"))
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo26n.pt"])
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--operating-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--gt-nms-iou", type=float, default=0.95)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def resize_frame(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return frame, 1.0
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale)))), scale


def scale_box(box: list[float], scale: float) -> list[float]:
    return [float(v) * scale for v in box]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def nms_boxes(boxes: list[list[float]], iou_threshold: float) -> list[list[float]]:
    ordered = sorted([box for box in boxes if box and box_area(box) > 0], key=box_area, reverse=True)
    kept: list[list[float]] = []
    for box in ordered:
        if all(box_iou(box, prev) < iou_threshold for prev in kept):
            kept.append(box)
    return kept


def match_boxes(
    preds: list[dict[str, Any]],
    gt_boxes: list[list[float]],
    iou_threshold: float,
    min_score: float = 0.0,
) -> tuple[int, int, int, list[float]]:
    preds = [pred for pred in preds if float(pred["score"]) >= min_score]
    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    tp = 0
    for pred in sorted(preds, key=lambda row: float(row["score"]), reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = box_iou(pred["box"], gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
            tp += 1
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn, matched_ious


def average_precision(rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    gt_by_frame = {(row["video_id"], row["frame"]): row["gt_boxes"] for row in rows}
    matched = {key: set() for key in gt_by_frame}
    total_gt = sum(len(boxes) for boxes in gt_by_frame.values())
    predictions = [
        {
            "video_id": row["video_id"],
            "frame": row["frame"],
            "score": pred["score"],
            "box": pred["box"],
        }
        for row in rows
        for pred in row["predictions"]
    ]
    predictions.sort(key=lambda pred: float(pred["score"]), reverse=True)
    if not predictions or total_gt == 0:
        return {"iou_threshold": iou_threshold, "ap": 0.0, "max_recall": 0.0, "evaluated_predictions": len(predictions)}

    tp_flags: list[float] = []
    fp_flags: list[float] = []
    for pred in predictions:
        key = (pred["video_id"], pred["frame"])
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_by_frame[key]):
            if idx in matched[key]:
                continue
            iou = box_iou(pred["box"], gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched[key].add(best_idx)
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)

    tp_cum = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))
    return {
        "iou_threshold": iou_threshold,
        "ap": ap,
        "max_recall": float(recalls[-1]) if recalls.size else 0.0,
        "evaluated_predictions": len(predictions),
    }


def read_frames(sample: dict[str, Any], root: Path, max_frames: int, max_side: int) -> list[dict[str, Any]]:
    video_path = root / sample["video"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    rows: list[dict[str, Any]] = []
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        objects = sample["frames"].get(str(frame_idx), [])
        if objects:
            frame, scale = resize_frame(frame, max_side)
            gt = nms_boxes(
                [scale_box(obj["bbox_xyxy"], scale) for obj in objects if obj.get("bbox_xyxy")],
                iou_threshold=0.95,
            )
            if gt:
                rows.append(
                    {
                        "video_id": sample["video_id"],
                        "frame": frame_idx,
                        "image": frame,
                        "gt_boxes": gt,
                    }
                )
        frame_idx += 1
    cap.release()
    return rows


def summarize(rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    tp = sum(int(row[f"tp_{iou_threshold}"]) for row in rows)
    fp = sum(int(row[f"fp_{iou_threshold}"]) for row in rows)
    fn = sum(int(row[f"fn_{iou_threshold}"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = [iou for row in rows for iou in row[f"matched_ious_{iou_threshold}"]]
    return {
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": statistics.fmean(matched) if matched else 0.0,
        "false_positives_per_frame": fp / len(rows) if rows else 0.0,
        "missed_gt_per_frame": fn / len(rows) if rows else 0.0,
    }


def run_model(model_path: str, frames: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    model = YOLO(model_path)
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in frames:
        started = time.perf_counter()
        result = model.predict(
            item["image"],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        latencies.append(time.perf_counter() - started)
        preds: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            preds = [
                {
                    "box": [float(v) for v in box],
                    "score": float(score),
                    "class_id": int(cls),
                }
                for box, score, cls in zip(boxes, confs, classes)
            ]
        row: dict[str, Any] = {
            "video_id": item["video_id"],
            "frame": item["frame"],
            "gt_boxes": item["gt_boxes"],
            "predictions": preds,
            "gt_count": len(item["gt_boxes"]),
            "pred_count": sum(1 for pred in preds if pred["score"] >= args.operating_conf),
            "raw_pred_count": len(preds),
            "latency_seconds": latencies[-1],
        }
        for threshold in (0.5, 0.75):
            tp, fp, fn, matched_ious = match_boxes(
                preds, item["gt_boxes"], threshold, min_score=args.operating_conf
            )
            row[f"tp_{threshold}"] = tp
            row[f"fp_{threshold}"] = fp
            row[f"fn_{threshold}"] = fn
            row[f"matched_ious_{threshold}"] = matched_ious
        rows.append(row)
    return {
        "model": model_path,
        "frames": len(rows),
        "gt_boxes": sum(row["gt_count"] for row in rows),
        "pred_boxes": sum(row["pred_count"] for row in rows),
        "raw_pred_boxes": sum(row["raw_pred_count"] for row in rows),
        "latency_ms_avg": statistics.fmean(latencies) * 1000 if latencies else 0.0,
        "latency_ms_p50": percentile(latencies, 50) * 1000,
        "latency_ms_p95": percentile(latencies, 95) * 1000,
        "fps": 1.0 / statistics.fmean(latencies) if latencies else 0.0,
        "metrics": [summarize(rows, threshold) for threshold in (0.5, 0.75)],
        "average_precision": [average_precision(rows, threshold) for threshold in (0.5, 0.75)],
        "per_frame": rows,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shard_root = args.manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    for sample in manifest["samples"][: args.max_samples]:
        frame_rows.extend(read_frames(sample, shard_root, args.max_frames, args.max_side))
    results = {
        "manifest": str(args.manifest),
        "samples": min(args.max_samples, len(manifest["samples"])),
        "evaluated_frames": len(frame_rows),
        "max_frames_per_sample": args.max_frames,
        "max_side": args.max_side,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "operating_conf": args.operating_conf,
        "nms_iou": args.iou,
        "gt_box_policy": "class-agnostic SA-V manual masklet boxes, duplicate GT boxes removed at IoU >= 0.95",
        "caveat": "SA-V is class-agnostic video object segmentation, not a sports-player detection benchmark. These metrics measure seed-box coverage on SA-V objects.",
        "models": [run_model(model_path, frame_rows, args) for model_path in args.models],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("model,frames,gt,pred@op,raw_pred,fps,p50_ms,p95_ms,ap50,ap75,p@50,r@50,f1@50,p@75,r@75,f1@75")
    for model in results["models"]:
        m50 = next(row for row in model["metrics"] if row["iou_threshold"] == 0.5)
        m75 = next(row for row in model["metrics"] if row["iou_threshold"] == 0.75)
        ap50 = next(row for row in model["average_precision"] if row["iou_threshold"] == 0.5)
        ap75 = next(row for row in model["average_precision"] if row["iou_threshold"] == 0.75)
        print(
            f"{model['model']},{model['frames']},{model['gt_boxes']},{model['pred_boxes']},{model['raw_pred_boxes']},"
            f"{model['fps']:.2f},{model['latency_ms_p50']:.2f},{model['latency_ms_p95']:.2f},"
            f"{ap50['ap']:.3f},{ap75['ap']:.3f},"
            f"{m50['precision']:.3f},{m50['recall']:.3f},{m50['f1']:.3f},"
            f"{m75['precision']:.3f},{m75['recall']:.3f},{m75['f1']:.3f}"
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
