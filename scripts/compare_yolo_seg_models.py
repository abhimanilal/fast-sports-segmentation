from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare YOLO segmentation models visually on sampled video frames.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[60, 120, 180, 240])
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=360)
    return parser.parse_args()


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video}")
    return frame


def overlay_result(frame: np.ndarray, model: YOLO, label: str, args: argparse.Namespace) -> np.ndarray:
    result = model.predict(frame[:, :, ::-1], imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
    out = frame.copy()
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
    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        for idx, (mask_small, box, conf) in enumerate(zip(masks, boxes, confs)):
            color = colors[idx % len(colors)]
            mask = cv2.resize(mask_small.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            out[mask] = (0.58 * out[mask] + 0.42 * color).astype(np.uint8)
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), color.tolist(), 2)
            cv2.putText(out, f"{conf:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 247, 242), 2, cv2.LINE_AA)
    return out


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.labels):
        raise ValueError("--models and --labels must have the same length")
    models = [YOLO(model_path) for model_path in args.models]
    rows = []
    for frame_idx in args.frames:
        frame = read_frame(args.video, frame_idx)
        cells = [resize_width(frame, args.width)]
        cv2.putText(cells[0], f"raw frame {frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 247, 242), 2, cv2.LINE_AA)
        for model, label in zip(models, args.labels):
            cells.append(resize_width(overlay_result(frame, model, label, args), args.width))
        max_h = max(cell.shape[0] for cell in cells)
        padded = [cv2.copyMakeBorder(cell, 0, max_h - cell.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(8, 12, 10)) for cell in cells]
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(args.output)


if __name__ == "__main__":
    main()
