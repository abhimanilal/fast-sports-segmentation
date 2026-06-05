from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleVideo:
    key: str
    sport: str
    filename: str
    url: str
    source_page: str
    license_note: str


SAMPLES = [
    SampleVideo(
        key="basketball_wheelchair",
        sport="basketball",
        filename="basketball_wheelchair.webm",
        url="https://commons.wikimedia.org/wiki/Special:Redirect/file/Wheelchair%20basketball.webm",
        source_page="https://commons.wikimedia.org/wiki/File:Wheelchair_basketball.webm",
        license_note="Wikimedia Commons; own work by Splash tv; CC license on file page.",
    ),
    SampleVideo(
        key="american_football_kickoff",
        sport="american_football",
        filename="american_football_kickoff.webm",
        url="https://commons.wikimedia.org/wiki/Special:Redirect/file/Kickoff%20Baker%20v%20Benedictine%202014.webm",
        source_page="https://commons.wikimedia.org/wiki/File:Kickoff_Baker_v_Benedictine_2014.webm",
        license_note="Wikimedia Commons; own work by Paulmcdonald; CC BY-SA 4.0.",
    ),
    SampleVideo(
        key="soccer_beautiful_game",
        sport="soccer",
        filename="soccer_beautiful_game.webm",
        url="https://commons.wikimedia.org/wiki/Special:Redirect/file/O%20Jogo%20Bonito%20%28The%20Beautiful%20Game%29.webm",
        source_page="https://commons.wikimedia.org/wiki/File:O_Jogo_Bonito_(The_Beautiful_Game).webm",
        license_note="Wikimedia Commons; own work by Sarah Samaha; CC BY-SA 4.0.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download license-friendly sports videos for local tracking tests."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--only", nargs="*", default=None, choices=[item.key for item in SAMPLES])
    return parser.parse_args()


def download(url: str, output: Path, retries: int = 4) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "fast-sports-segmentation/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                output.write_bytes(response.read())
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            wait_seconds = 15 * (attempt + 1)
            print(f"rate limited; waiting {wait_seconds}s before retry")
            time.sleep(wait_seconds)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [item for item in SAMPLES if args.only is None or item.key in args.only]
    manifest = []
    for item in selected:
        output = args.output_dir / item.filename
        if not output.exists() or output.stat().st_size == 0:
            print(f"downloading {item.key}: {item.source_page}")
            download(item.url, output)
        else:
            print(f"exists {item.key}: {output}")
        time.sleep(1.5)
        record = asdict(item)
        record["path"] = str(output)
        record["bytes"] = output.stat().st_size
        manifest.append(record)
    manifest_path = args.output_dir / "sample_videos_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
