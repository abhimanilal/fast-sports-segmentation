from __future__ import annotations

import argparse
import json
import re
import time
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LocateAnything boxes -> EfficientSAM3 box-prompt masks on sampled video frames."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detect_then_segment"))
    parser.add_argument("--categories", nargs="+", default=["baseball player"])
    parser.add_argument("--locate-model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--boxes-json", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--sam-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--locate-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sam-dtype",
        default="fp16" if torch.cuda.is_available() else "fp32",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--locate-dtype",
        default="fp16" if torch.cuda.is_available() else "fp32",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--locate-load-in-8bit", action="store_true")
    parser.add_argument("--locate-max-gpu-memory", default="5200MiB")
    parser.add_argument("--locate-max-cpu-memory", default="8GiB")
    parser.add_argument("--locate-offload-folder", default="E:/HFOffload/LocateAnything")
    parser.add_argument("--generation-mode", default="fast", choices=["fast", "hybrid", "slow"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--box-pad-frac", type=float, default=0.04)
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


def load_boxes_json(path: Path) -> dict[str, list[list[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): value for key, value in data.items()}


def pad_and_clip_boxes(
    boxes: list[list[float]], width: int, height: int, pad_frac: float
) -> np.ndarray:
    padded = []
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        pad_x = (x2 - x1) * pad_frac
        pad_y = (y2 - y1) * pad_frac
        padded.append(
            [
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(width - 1), x2 + pad_x),
                min(float(height - 1), y2 + pad_y),
            ]
        )
    return np.asarray(padded, dtype=np.float32)


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


def overlay(frame_rgb: np.ndarray, masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = frame_rgb.copy()
    colors = np.asarray(
        [[245, 90, 40], [30, 160, 220], [80, 200, 120], [230, 190, 50]],
        dtype=np.uint8,
    )
    for idx, mask in enumerate(masks):
        mask_bool = np.squeeze(mask).astype(bool)
        color = colors[idx % len(colors)]
        out[mask_bool] = (0.55 * out[mask_bool] + 0.45 * color).astype(np.uint8)
        x1, y1, x2, y2 = boxes[idx].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color.tolist(), 2)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sam_dtype = resolve_dtype(args.sam_dtype, args.sam_device)
    locate_dtype = resolve_dtype(args.locate_dtype, args.locate_device)

    boxes_by_frame = load_boxes_json(args.boxes_json) if args.boxes_json else {}
    detector = (
        None
        if boxes_by_frame
        else LocateAnythingDetector(
            args.locate_model,
            args.locate_device,
            locate_dtype,
            args.locate_load_in_8bit,
            args.locate_max_gpu_memory,
            args.locate_max_cpu_memory,
            args.locate_offload_folder,
        )
    )

    sam_model = build_efficientsam3_image_model(
        checkpoint_path=str(args.sam_checkpoint),
        backbone_type="efficientvit",
        model_name="b0",
        text_encoder_type="MobileCLIP-S1",
        text_encoder_context_length=77,
        enable_text_encoder=False,
        enable_inst_interactivity=True,
        device=args.sam_device,
    )
    if args.sam_device == "cuda" and sam_dtype != torch.float32:
        sam_model = sam_model.to(dtype=sam_dtype)
    processor = Sam3Processor(sam_model, device=args.sam_device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    metrics: dict[str, Any] = {
        "video": str(args.video),
        "sam_checkpoint": str(args.sam_checkpoint),
        "categories": args.categories,
        "sam_dtype": str(sam_dtype).replace("torch.", ""),
        "locate_dtype": str(locate_dtype).replace("torch.", ""),
        "frames": [],
    }
    last_boxes: list[list[float]] = []
    processed = 0
    frame_index = 0

    while processed < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb, scale = resize_rgb(frame_bgr, args.max_side)
        image = Image.fromarray(frame_rgb)

        detect_start = time.perf_counter()
        if str(frame_index) in boxes_by_frame:
            boxes = boxes_by_frame[str(frame_index)]
            answer = "boxes-json"
        elif detector is not None and processed % args.detect_every == 0:
            answer, boxes = detector.detect(
                image, args.categories, args.generation_mode, args.max_new_tokens
            )
            last_boxes = boxes
        else:
            boxes = last_boxes
            answer = "reused-last-boxes"
        detect_seconds = time.perf_counter() - detect_start

        box_array = pad_and_clip_boxes(
            boxes[: args.max_boxes], image.width, image.height, args.box_pad_frac
        )
        if len(box_array) == 0:
            metrics["frames"].append(
                {"frame": frame_index, "boxes": [], "answer": answer, "scale": scale}
            )
            processed += 1
            frame_index += args.stride
            continue

        seg_start = time.perf_counter()
        with torch.inference_mode(), autocast_context(args.sam_device, sam_dtype):
            state = processor.set_image(image)
            masks, scores, _ = sam_model.predict_inst(
                state, box=box_array, multimask_output=False
            )
            if args.sam_device == "cuda":
                torch.cuda.synchronize()
        seg_seconds = time.perf_counter() - seg_start

        masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
        scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        vis = overlay(frame_rgb, masks_np, box_array)
        out_path = args.output_dir / f"frame_{frame_index:05d}_boxes_masks.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        metrics["frames"].append(
            {
                "frame": frame_index,
                "output": str(out_path),
                "answer": answer,
                "boxes": box_array.tolist(),
                "scores": scores_np.reshape(-1).astype(float).tolist(),
                "detect_seconds": detect_seconds,
                "segment_seconds": seg_seconds,
                "scale": scale,
            }
        )
        processed += 1
        frame_index += args.stride

    cap.release()
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
