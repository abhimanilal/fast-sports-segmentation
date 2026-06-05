from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


POSITIVE_PROMPTS = [
    "basketball court with a hoop and players",
    "indoor basketball court wooden gym floor",
    "people playing basketball on a court",
    "sports players running during a game on a court",
    "soccer field or tennis court with players",
]

NEGATIVE_PROMPTS = [
    "a close up selfie",
    "a person cooking in a kitchen",
    "a pet or animal video",
    "a car driving on a road",
    "an object on a table",
    "a person sitting at home",
    "people standing in a shopping mall",
    "people waiting in line indoors",
    "an airport or train station crowd",
    "people dancing on a stage",
    "a store interior with shoppers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Rank SA-V Subset 51 videos for sports-player relevance using CLIP.")
    parser.add_argument("--repo-id", default="Voxel51/segment_anything_video_subset51")
    parser.add_argument(
        "--hub-root",
        type=Path,
        default=Path(r"E:\FiftyOne\huggingface\hub\Voxel51\segment_anything_video_subset51"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/sav_sports_candidates"))
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--prefer-indoor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def candidate_samples(samples: list[dict[str, Any]], max_candidates: int, prefer_indoor: bool) -> list[dict[str, Any]]:
    def score_meta(sample: dict[str, Any]) -> tuple[float, str]:
        metadata = sample.get("metadata", {})
        width = float(metadata.get("frame_width") or 1)
        height = float(metadata.get("frame_height") or 1)
        aspect = max(width, height) / max(1.0, min(width, height))
        manual = float(sample.get("num_manual_masklets") or 0)
        auto = float(sample.get("num_auto_masklets") or 0)
        duration = float(sample.get("video_duration") or 0)
        indoor_bonus = 1.0 if sample.get("video_environment") == "Indoor" else 0.0
        mask_score = min(manual, 8.0) + 0.15 * min(auto, 20.0)
        duration_score = 1.0 if 10.0 <= duration <= 18.0 else 0.0
        aspect_score = 1.0 if aspect <= 1.9 else 0.35
        env_score = indoor_bonus if prefer_indoor else 0.5
        return (mask_score + duration_score + aspect_score + env_score, sample["video_id"])

    ranked = sorted(samples, key=score_meta, reverse=True)
    return ranked[:max_candidates]


def download_video(repo_id: str, sample: dict[str, Any], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = sample["filepath"].replace("\\", "/")
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=cache_dir)
    return Path(path)


def sample_video_frames(video_path: Path, count: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    frame_indices = np.linspace(max(0, total * 0.15), max(0, total * 0.85), count, dtype=int)
    frames: list[Image.Image] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def score_frames(
    model: CLIPModel,
    processor: CLIPProcessor,
    frames: list[Image.Image],
    prompt_features: torch.Tensor,
    device: str,
) -> tuple[float, float, float]:
    if not frames:
        return -999.0, -999.0, 999.0
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ prompt_features.T
    pos = sims[:, : len(POSITIVE_PROMPTS)].max(dim=1).values
    neg = sims[:, len(POSITIVE_PROMPTS) :].max(dim=1).values
    margins = pos - neg
    return float(margins.mean().item()), float(pos.mean().item()), float(neg.mean().item())


def make_contact_sheet(rows: list[dict[str, Any]], output: Path, thumb_width: int = 240) -> None:
    thumbs = []
    for row in rows:
        video = Path(row["local_path"])
        frames = sample_video_frames(video, 1)
        if not frames:
            continue
        image = cv2.cvtColor(np.asarray(frames[0]), cv2.COLOR_RGB2BGR)
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.putText(
            thumb,
            f"{row['rank']:02d} {row['video_id']} {row['score']:.3f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    if not thumbs:
        return
    rows_out = []
    for idx in range(0, len(thumbs), 4):
        cells = thumbs[idx : idx + 4]
        max_h = max(cell.shape[0] for cell in cells)
        cells = [cv2.copyMakeBorder(cell, 0, max_h - cell.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(8, 12, 10)) for cell in cells]
        rows_out.append(np.hstack(cells))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows_out))


def main() -> None:
    args = parse_args()
    samples_path = args.hub_root / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))["samples"]
    candidates = candidate_samples(samples, args.max_candidates, args.prefer_indoor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "hf_cache"

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(args.device).eval()
    text_inputs = processor(text=POSITIVE_PROMPTS + NEGATIVE_PROMPTS, return_tensors="pt", padding=True).to(args.device)
    with torch.inference_mode():
        prompt_features = model.get_text_features(**text_inputs)
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    scored = []
    for idx, sample in enumerate(candidates, start=1):
        local_path = download_video(args.repo_id, sample, cache_dir)
        frames = sample_video_frames(local_path, args.frames_per_video)
        score, pos, neg = score_frames(model, processor, frames, prompt_features, args.device)
        row = {
            "video_id": sample["video_id"],
            "filepath": sample["filepath"],
            "local_path": str(local_path),
            "score": score,
            "positive_score": pos,
            "negative_score": neg,
            "video_environment": sample.get("video_environment"),
            "num_manual_masklets": sample.get("num_manual_masklets"),
            "num_auto_masklets": sample.get("num_auto_masklets"),
            "duration": sample.get("video_duration"),
        }
        scored.append(row)
        print(f"{idx:03d}/{len(candidates)} {row['video_id']} score={score:.4f} pos={pos:.4f} neg={neg:.4f}")

    scored.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    selected = scored[: args.top_k]
    (args.output_dir / "ranked_candidates.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")
    (args.output_dir / "selected_ids.txt").write_text("\n".join(row["video_id"] for row in selected) + "\n", encoding="utf-8")
    make_contact_sheet(selected, args.output_dir / "selected_contact_sheet.jpg")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
