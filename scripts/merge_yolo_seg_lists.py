from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge YOLO segmentation list datasets without copying images.")
    parser.add_argument("--primary", type=Path, required=True, help="Primary YOLO dataset directory.")
    parser.add_argument("--aux", type=Path, required=True, help="Auxiliary YOLO dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aux-limit", type=int, default=None)
    parser.add_argument("--aux-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-source", choices=["primary", "aux"], default="primary")
    parser.add_argument("--class-name", default="player")
    return parser.parse_args()


def read_list(dataset_dir: Path, split: str) -> list[str]:
    list_path = dataset_dir / f"{split}.txt"
    if not list_path.exists():
        raise FileNotFoundError(list_path)
    return [line.strip().replace("\\", "/") for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    primary_train = read_list(args.primary, "train")
    aux_train_base = read_list(args.aux, "train")
    if args.aux_limit is not None and args.aux_limit < len(aux_train_base):
        rng = random.Random(args.seed)
        aux_train_base = sorted(rng.sample(aux_train_base, args.aux_limit))
    aux_train = aux_train_base * args.aux_repeat
    val_dir = args.primary if args.val_source == "primary" else args.aux
    val = read_list(val_dir, "val")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text("\n".join(primary_train + aux_train) + "\n", encoding="utf-8")
    (args.output_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    data = {
        "path": str(Path.cwd()).replace("\\", "/"),
        "train": str((args.output_dir / "train.txt").resolve()).replace("\\", "/"),
        "val": str((args.output_dir / "val.txt").resolve()).replace("\\", "/"),
        "names": {0: args.class_name},
    }
    (args.output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(
        {
            "output_dir": str(args.output_dir),
            "primary_train": len(primary_train),
            "aux_train": len(aux_train),
            "aux_limit": args.aux_limit,
            "val": len(val),
            "aux_repeat": args.aux_repeat,
            "val_source": args.val_source,
        }
    )


if __name__ == "__main__":
    main()
