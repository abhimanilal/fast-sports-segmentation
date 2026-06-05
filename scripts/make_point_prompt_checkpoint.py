from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip an EfficientSAM3 model-zoo checkpoint down to point-prompt inference weights."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.input, map_location="cpu", weights_only=True, mmap=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    keep_prefixes = (
        "detector.backbone.vision_backbone.",
        "tracker.",
    )
    slim = {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
        if key.startswith(keep_prefixes)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": slim}, args.output)

    input_mb = args.input.stat().st_size / 1024 / 1024
    output_mb = args.output.stat().st_size / 1024 / 1024
    print(
        f"kept {len(slim)} tensors; {input_mb:.1f} MiB -> {output_mb:.1f} MiB: {args.output}"
    )


if __name__ == "__main__":
    main()
