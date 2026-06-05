from __future__ import annotations

import argparse
import json
import re
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    template: np.ndarray
    score: float = 1.0
    age: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track detector boxes through video and refresh masks with vision-only EfficientSAM3."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracking"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--sam-every", type=int, default=5)
    parser.add_argument("--search-pad", type=float, default=1.6)
    parser.add_argument("--template-update", type=float, default=0.08)
    parser.add_argument("--mask-update-threshold", type=float, default=0.55)
    parser.add_argument("--mask-box-pad-frac", type=float, default=0.08)
    parser.add_argument("--yolo-every", type=int, default=0)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--sam-on-yolo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--locate-every", type=int, default=0)
    parser.add_argument("--locate-model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--locate-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--locate-dtype",
        default="fp16" if torch.cuda.is_available() else "fp32",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--locate-load-in-8bit", action="store_true")
    parser.add_argument("--locate-max-gpu-memory", default="5200MiB")
    parser.add_argument("--locate-max-cpu-memory", default="8GiB")
    parser.add_argument("--locate-offload-folder", default="E:/HFOffload/LocateAnything")
    parser.add_argument("--categories", nargs="+", default=["person"])
    parser.add_argument("--generation-mode", default="fast", choices=["fast", "hybrid", "slow"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--box-nms-iou", type=float, default=0.45)
    parser.add_argument("--min-box-area-frac", type=float, default=0.0015)
    parser.add_argument("--max-box-area-frac", type=float, default=0.22)
    parser.add_argument("--min-box-side", type=float, default=8.0)
    parser.add_argument("--min-box-aspect", type=float, default=0.12)
    parser.add_argument("--max-box-aspect", type=float, default=1.25)
    parser.add_argument("--edge-margin-frac", type=float, default=0.01)
    parser.add_argument("--admit-sam-threshold", type=float, default=0.30)
    parser.add_argument("--prune-sam-threshold", type=float, default=0.20)
    parser.add_argument("--drop-edge-tracks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="fp16" if torch.cuda.is_available() else "fp32",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--write-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-empty-tracks", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def autocast_context(device: str, dtype: torch.dtype):
    enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def resize_rgb(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame_rgb, scale


def load_boxes(path: Path | None) -> dict[int, list[list[float]]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(frame): boxes for frame, boxes in raw.items()}


def resolve_input_boxes(boxes: list[list[float]], width: int, height: int) -> list[list[float]]:
    if not boxes:
        return []
    flat = [coord for box in boxes for coord in box]
    if flat and min(flat) >= 0.0 and max(flat) <= 1.0:
        return [
            [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            for box in boxes
        ]
    return boxes


def parse_locate_boxes(answer: str, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            [
                x1 / 1000.0 * width,
                y1 / 1000.0 * height,
                x2 / 1000.0 * width,
                y2 / 1000.0 * height,
            ]
        )
    return boxes


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.astype(float)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_overlap_min(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(box_area(a), box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def is_usable_box(
    box: np.ndarray,
    width: int,
    height: int,
    min_area_frac: float,
    max_area_frac: float,
    min_side: float,
    min_aspect: float,
    max_aspect: float,
    edge_margin_frac: float,
    reject_edge: bool,
) -> bool:
    x1, y1, x2, y2 = box.astype(float)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    area_frac = (bw * bh) / float(width * height)
    aspect = bw / bh
    if (
        bw < min_side
        or bh < min_side
        or area_frac < min_area_frac
        or area_frac > max_area_frac
        or aspect < min_aspect
        or aspect > max_aspect
    ):
        return False
    edge_margin = min(width, height) * edge_margin_frac
    if reject_edge and (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= width - 1 - edge_margin
        or y2 >= height - 1 - edge_margin
    ):
        return False
    return True


def filter_detected_boxes(
    boxes: list[list[float]],
    width: int,
    height: int,
    max_detections: int,
    nms_iou: float,
    min_area_frac: float,
    max_area_frac: float,
    min_side: float,
    min_aspect: float,
    max_aspect: float,
    edge_margin_frac: float,
) -> tuple[list[list[float]], dict[str, int]]:
    stats = {
        "raw": len(boxes),
        "invalid": 0,
        "shape": 0,
        "edge": 0,
        "nms": 0,
        "kept": 0,
    }
    frame_area = float(width * height)
    edge_margin = min(width, height) * edge_margin_frac
    candidates: list[np.ndarray] = []
    for box in boxes:
        clipped = clip_box(np.asarray(box, dtype=np.float32), width, height)
        x1, y1, x2, y2 = clipped.astype(float)
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            stats["invalid"] += 1
            continue
        reject_edge = (
            x1 <= edge_margin
            or y1 <= edge_margin
            or x2 >= width - 1 - edge_margin
            or y2 >= height - 1 - edge_margin
        )
        if reject_edge:
            stats["edge"] += 1
            continue
        if not is_usable_box(
            clipped,
            width,
            height,
            min_area_frac,
            max_area_frac,
            min_side,
            min_aspect,
            max_aspect,
            edge_margin_frac,
            reject_edge=False,
        ):
            stats["shape"] += 1
            continue
        candidates.append(clipped)

    # With no detector confidences, favor plausible full-body player boxes:
    # larger boxes first, but after size/shape/edge filters have removed broad junk.
    candidates.sort(key=box_area, reverse=True)
    kept: list[np.ndarray] = []
    for candidate in candidates:
        if any(
            box_iou(candidate, existing) > nms_iou
            or box_overlap_min(candidate, existing) > 0.82
            for existing in kept
        ):
            stats["nms"] += 1
            continue
        kept.append(candidate)
        if len(kept) >= max_detections:
            break
    stats["kept"] = len(kept)
    return [box.astype(float).tolist() for box in kept], stats


class LocateAnythingDetector:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: torch.dtype,
        load_in_8bit: bool,
        max_gpu_memory: str,
        max_cpu_memory: str,
        offload_folder: str,
    ):
        import transformers.modeling_utils as modeling_utils
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "max_memory": {0: max_gpu_memory, "cpu": max_cpu_memory},
            "offload_folder": offload_folder,
            "offload_state_dict": True,
        }
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name, **kwargs).eval()
        language_core = getattr(getattr(self.model, "language_model", None), "model", None)
        if language_core is not None and hasattr(language_core, "image_processing"):
            def image_processing_cast(module_self, input_ids, visual_features, image_token_index):
                if visual_features is not None:
                    input_embeds = module_self.get_input_embeddings()(input_ids)
                    batch, seq_len, channels = input_embeds.shape
                    flat_embeds = input_embeds.reshape(batch * seq_len, channels)
                    flat_ids = input_ids.reshape(batch * seq_len)
                    selected = flat_ids == image_token_index
                    assert selected.sum() != 0
                    flat_embeds[selected] = visual_features.reshape(-1, channels).to(
                        device=flat_embeds.device, dtype=flat_embeds.dtype
                    )
                    return flat_embeds.reshape(batch, seq_len, channels)
                return module_self.get_input_embeddings()(input_ids)

            language_core.image_processing = types.MethodType(
                image_processing_cast, language_core
            )

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        categories: list[str],
        generation_mode: str,
        max_new_tokens: int,
    ) -> tuple[str, list[list[float]]]:
        cats = "</c>".join(categories)
        question = f"Locate all the instances that matches the following description: {cats}."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=0.1,
            do_sample=False,
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return answer, parse_locate_boxes(answer, image.width, image.height)


class YoloPersonDetector:
    def __init__(self, model_name: str, device: str, conf: float, imgsz: int):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame_rgb: np.ndarray) -> list[list[float]]:
        results = self.model.predict(
            frame_rgb,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes: list[list[float]] = []
        if not results:
            return boxes
        result = results[0]
        if result.boxes is None:
            return boxes
        for xyxy in result.boxes.xyxy.detach().cpu().numpy():
            boxes.append([float(value) for value in xyxy])
        return boxes


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    return np.asarray(
        [
            np.clip(x1, 0, width - 2),
            np.clip(y1, 0, height - 2),
            np.clip(x2, 1, width - 1),
            np.clip(y2, 1, height - 1),
        ],
        dtype=np.float32,
    )


def crop_gray(gray: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return gray[y1:y2, x1:x2].copy()


def mask_to_box(mask: np.ndarray, width: int, height: int, pad_frac: float) -> np.ndarray | None:
    ys, xs = np.where(np.squeeze(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    pad_x = (x2 - x1) * pad_frac
    pad_y = (y2 - y1) * pad_frac
    return clip_box(np.asarray([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]), width, height)


def init_tracks(gray: np.ndarray, boxes: list[list[float]], width: int, height: int) -> list[Track]:
    tracks: list[Track] = []
    for idx, box in enumerate(boxes):
        clipped = clip_box(np.asarray(box, dtype=np.float32), width, height)
        template = crop_gray(gray, clipped)
        if template.size == 0:
            continue
        tracks.append(Track(track_id=idx, box=clipped, template=template))
    return tracks


def search_region(box: np.ndarray, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(float)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sw, sh = bw * (1 + pad), bh * (1 + pad)
    sx1 = int(max(0, cx - sw / 2))
    sy1 = int(max(0, cy - sh / 2))
    sx2 = int(min(width, cx + sw / 2))
    sy2 = int(min(height, cy + sh / 2))
    return sx1, sy1, sx2, sy2


def update_track(gray: np.ndarray, track: Track, search_pad: float, template_update: float) -> None:
    h, w = gray.shape
    sx1, sy1, sx2, sy2 = search_region(track.box, w, h, search_pad)
    search = gray[sy1:sy2, sx1:sx2]
    template = track.template
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        track.score = 0.0
        track.age += 1
        return
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x1 = sx1 + max_loc[0]
    y1 = sy1 + max_loc[1]
    x2 = x1 + template.shape[1]
    y2 = y1 + template.shape[0]
    track.box = clip_box(np.asarray([x1, y1, x2, y2], dtype=np.float32), w, h)
    new_template = crop_gray(gray, track.box)
    if new_template.shape == template.shape and max_val > 0.25:
        track.template = cv2.addWeighted(template, 1.0 - template_update, new_template, template_update, 0)
    track.score = float(max_val)
    track.age += 1


def update_tracks_from_masks(
    gray: np.ndarray,
    tracks: list[Track],
    masks: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    pad_frac: float,
) -> None:
    height, width = gray.shape
    for idx, track in enumerate(tracks):
        if idx >= len(masks) or idx >= len(scores) or float(scores[idx]) < score_threshold:
            continue
        mask_box = mask_to_box(masks[idx], width, height, pad_frac)
        if mask_box is None:
            continue
        template = crop_gray(gray, mask_box)
        if template.size == 0:
            continue
        track.box = mask_box
        track.template = template


def prune_tracks_after_sam(
    tracks: list[Track],
    masks: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    score_threshold: float,
    min_area_frac: float,
    max_area_frac: float,
    min_side: float,
    min_aspect: float,
    max_aspect: float,
    edge_margin_frac: float,
    drop_edge_tracks: bool,
) -> tuple[list[Track], np.ndarray, np.ndarray, dict[str, int]]:
    stats = {"before": len(tracks), "low_sam": 0, "shape": 0, "edge": 0, "after": 0}
    keep_indices: list[int] = []
    for idx, track in enumerate(tracks):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score < score_threshold:
            stats["low_sam"] += 1
            continue
        if not is_usable_box(
            track.box,
            width,
            height,
            min_area_frac,
            max_area_frac,
            min_side,
            min_aspect,
            max_aspect,
            edge_margin_frac,
            reject_edge=False,
        ):
            stats["shape"] += 1
            continue
        if drop_edge_tracks and not is_usable_box(
            track.box,
            width,
            height,
            min_area_frac,
            max_area_frac,
            min_side,
            min_aspect,
            max_aspect,
            edge_margin_frac,
            reject_edge=True,
        ):
            stats["edge"] += 1
            continue
        keep_indices.append(idx)
    stats["after"] = len(keep_indices)
    return (
        [tracks[idx] for idx in keep_indices],
        masks[keep_indices] if keep_indices else np.asarray([], dtype=masks.dtype),
        scores[keep_indices] if keep_indices else np.asarray([], dtype=scores.dtype),
        stats,
    )


def sam_segment(
    model,
    processor,
    image: Image.Image,
    boxes: np.ndarray,
    device: str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, dtype):
        state = processor.set_image(image)
        masks, scores, _ = model.predict_inst(state, box=boxes, multimask_output=False)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    return masks_np, scores_np.reshape(-1), elapsed


def overlay(frame_rgb: np.ndarray, tracks: list[Track], masks: np.ndarray | None) -> np.ndarray:
    out = frame_rgb.copy()
    colors = np.asarray(
        [[245, 90, 40], [30, 160, 220], [80, 200, 120], [230, 190, 50]],
        dtype=np.uint8,
    )
    if masks is not None:
        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            mask_bool = np.squeeze(mask).astype(bool)
            out[mask_bool] = (0.55 * out[mask_bool] + 0.45 * color).astype(np.uint8)
    for idx, track in enumerate(tracks):
        color = colors[idx % len(colors)].tolist()
        x1, y1, x2, y2 = track.box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"id={track.track_id} t={track.score:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype, args.device)
    locate_dtype = resolve_dtype(args.locate_dtype, args.locate_device)
    boxes_by_frame = load_boxes(args.boxes_json)
    yolo_detector = (
        YoloPersonDetector(args.yolo_model, args.yolo_device, args.yolo_conf, args.yolo_imgsz)
        if args.yolo_every > 0
        else None
    )
    detector = (
        LocateAnythingDetector(
            args.locate_model,
            args.locate_device,
            locate_dtype,
            args.locate_load_in_8bit,
            args.locate_max_gpu_memory,
            args.locate_max_cpu_memory,
            args.locate_offload_folder,
        )
        if args.locate_every > 0
        else None
    )

    model = build_efficientsam3_image_model(
        checkpoint_path=str(args.sam_checkpoint),
        backbone_type="efficientvit",
        model_name="b0",
        enable_text_encoder=False,
        enable_inst_interactivity=True,
        device=args.device,
    )
    if args.device == "cuda" and dtype != torch.float32:
        model = model.to(dtype=dtype)
    processor = Sam3Processor(model, device=args.device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    tracks: list[Track] = []
    writer = None
    metrics: dict[str, Any] = {
        "video": str(args.video),
        "sam_checkpoint": str(args.sam_checkpoint),
        "dtype": str(dtype).replace("torch.", ""),
        "yolo_every": args.yolo_every,
        "yolo_model": args.yolo_model,
        "locate_every": args.locate_every,
        "locate_dtype": str(locate_dtype).replace("torch.", ""),
        "categories": args.categories,
        "source_fps": source_fps,
        "frames": [],
    }

    for local_idx in range(args.max_frames):
        frame_start = time.perf_counter()
        frame_idx = args.start_frame + local_idx
        read_start = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        read_seconds = time.perf_counter() - read_start
        preprocess_start = time.perf_counter()
        frame_rgb, scale = resize_rgb(frame_bgr, args.max_side)
        image = Image.fromarray(frame_rgb)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        preprocess_seconds = time.perf_counter() - preprocess_start

        detection_source = None
        detection_answer = None
        box_filter_stats = None
        admitted_track_ids = None
        track_prune_stats = None
        locate_seconds = 0.0
        track_seconds = 0.0
        detection_seconds = 0.0
        if frame_idx in boxes_by_frame:
            detection_start = time.perf_counter()
            input_boxes = resolve_input_boxes(boxes_by_frame[frame_idx], w, h)
            filtered_boxes, box_filter_stats = filter_detected_boxes(
                input_boxes,
                w,
                h,
                args.max_detections,
                args.box_nms_iou,
                args.min_box_area_frac,
                args.max_box_area_frac,
                args.min_box_side,
                args.min_box_aspect,
                args.max_box_aspect,
                args.edge_margin_frac,
            )
            tracks = init_tracks(gray, filtered_boxes, w, h)
            detection_source = "boxes-json"
            detection_seconds = time.perf_counter() - detection_start
        elif yolo_detector is not None and (
            local_idx == 0 or local_idx % args.yolo_every == 0 or not tracks
        ):
            detection_start = time.perf_counter()
            detected_boxes = yolo_detector.detect(frame_rgb)
            filtered_boxes, box_filter_stats = filter_detected_boxes(
                detected_boxes,
                w,
                h,
                args.max_detections,
                args.box_nms_iou,
                args.min_box_area_frac,
                args.max_box_area_frac,
                args.min_box_side,
                args.min_box_aspect,
                args.max_box_aspect,
                args.edge_margin_frac,
            )
            if filtered_boxes:
                tracks = init_tracks(gray, filtered_boxes, w, h)
                detection_source = "yolo"
            elif not tracks and not args.allow_empty_tracks:
                raise RuntimeError(
                    f"YOLO found no usable person boxes on frame {frame_idx}; "
                    "seed with --boxes-json or adjust --yolo-conf."
                )
            detection_seconds = time.perf_counter() - detection_start
        elif detector is not None and (
            local_idx == 0 or local_idx % args.locate_every == 0 or not tracks
        ):
            locate_start = time.perf_counter()
            detection_answer, detected_boxes = detector.detect(
                image, args.categories, args.generation_mode, args.max_new_tokens
            )
            locate_seconds = time.perf_counter() - locate_start
            detection_start = time.perf_counter()
            filtered_boxes, box_filter_stats = filter_detected_boxes(
                detected_boxes,
                w,
                h,
                args.max_detections,
                args.box_nms_iou,
                args.min_box_area_frac,
                args.max_box_area_frac,
                args.min_box_side,
                args.min_box_aspect,
                args.max_box_aspect,
                args.edge_margin_frac,
            )
            if filtered_boxes:
                tracks = init_tracks(gray, filtered_boxes, w, h)
                detection_source = "locate-anything"
            elif not tracks:
                raise RuntimeError(
                    f"LocateAnything found no usable boxes on frame {frame_idx}; "
                    "seed with --boxes-json or adjust --categories."
                )
            detection_seconds = locate_seconds + (time.perf_counter() - detection_start)
        elif not tracks:
            if not args.allow_empty_tracks:
                raise RuntimeError(f"No seed boxes available for frame {frame_idx}")
        else:
            track_start = time.perf_counter()
            for track in tracks:
                update_track(gray, track, args.search_pad, args.template_update)
            track_seconds = time.perf_counter() - track_start

        masks = None
        sam_scores: list[float] = []
        sam_seconds = 0.0
        should_refresh_masks = (
            local_idx % args.sam_every == 0
            or detection_source in {"locate-anything", "boxes-json"}
            or (detection_source == "yolo" and args.sam_on_yolo)
        )
        if tracks and should_refresh_masks:
            boxes = np.asarray([track.box for track in tracks], dtype=np.float32)
            masks, scores, sam_seconds = sam_segment(
                model, processor, image, boxes, args.device, dtype
            )
            if detection_source in {"locate-anything", "boxes-json"} or (
                detection_source == "yolo" and args.sam_on_yolo
            ):
                keep_indices = [
                    idx for idx, score in enumerate(scores) if float(score) >= args.admit_sam_threshold
                ]
                admitted_track_ids = [tracks[idx].track_id for idx in keep_indices]
                tracks = [tracks[idx] for idx in keep_indices]
                masks = masks[keep_indices] if keep_indices else None
                scores = scores[keep_indices] if keep_indices else np.asarray([], dtype=np.float32)
            update_tracks_from_masks(
                gray,
                tracks,
                masks,
                scores,
                args.mask_update_threshold,
                args.mask_box_pad_frac,
            )
            tracks, masks, scores, track_prune_stats = prune_tracks_after_sam(
                tracks,
                masks,
                scores,
                w,
                h,
                args.prune_sam_threshold,
                args.min_box_area_frac,
                args.max_box_area_frac,
                args.min_box_side,
                args.min_box_aspect,
                args.max_box_aspect,
                args.edge_margin_frac,
                args.drop_edge_tracks,
            )
            sam_scores = scores.astype(float).tolist()

        render_start = time.perf_counter()
        vis = overlay(frame_rgb, tracks, masks) if args.render_overlays else frame_rgb
        render_seconds = time.perf_counter() - render_start

        write_seconds = 0.0
        frame_path = None
        if args.write_frames:
            write_start = time.perf_counter()
            frame_path = args.output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            write_seconds += time.perf_counter() - write_start

        if args.write_video:
            if writer is None:
                video_path = args.output_dir / "tracked.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, source_fps, (w, h))
            write_start = time.perf_counter()
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            write_seconds += time.perf_counter() - write_start

        frame_seconds = time.perf_counter() - frame_start

        metrics["frames"].append(
            {
                "frame": frame_idx,
                "output": str(frame_path) if frame_path is not None else None,
                "boxes": [track.box.astype(float).tolist() for track in tracks],
                "track_scores": [track.score for track in tracks],
                "sam_scores": sam_scores,
                "sam_seconds": sam_seconds,
                "read_seconds": read_seconds,
                "preprocess_seconds": preprocess_seconds,
                "detection_seconds": detection_seconds,
                "track_seconds": track_seconds,
                "render_seconds": render_seconds,
                "write_seconds": write_seconds,
                "frame_seconds": frame_seconds,
                "detection_source": detection_source,
                "box_filter_stats": box_filter_stats,
                "admitted_track_ids": admitted_track_ids,
                "track_prune_stats": track_prune_stats,
                "locate_seconds": locate_seconds,
                "locate_answer": detection_answer,
                "scale": scale,
            }
        )

    cap.release()
    if writer is not None:
        writer.release()
        metrics["video_output"] = str(args.output_dir / "tracked.mp4")
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.print_metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
