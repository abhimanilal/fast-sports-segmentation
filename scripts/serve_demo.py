from __future__ import annotations

import cgi
import json
import mimetypes
import os
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "data" / "uploads"
SUMMARY_PATH = ROOT / "outputs" / "portfolio_report" / "summary.json"
PUBLIC_SUMMARY_PATH = ROOT / "demo_site" / "summary.json"
DEFAULT_VIDEO = ROOT / "data" / "raw" / "youtube" / "clips" / "rec_league_0008_45s.mp4"
PUBLIC_VIDEO = ROOT / "media" / "sports_segmentation_demo_15s.mp4"


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "FastSportsDemo/0.1"

    def translate_path(self, path: str) -> str:
        clean = unquote(path.split("?", 1)[0].split("#", 1)[0]).lstrip("/")
        if clean == "":
            clean = "demo_site/index.html"
        requested = Path(clean)
        if requested.is_absolute() or ".." in requested.parts:
            return str(ROOT / "demo_site" / "index.html")
        return str(ROOT / requested)

    def do_GET(self) -> None:
        if self.path == "/api/summary":
            summary_path = SUMMARY_PATH if SUMMARY_PATH.exists() else PUBLIC_SUMMARY_PATH
            payload = read_json(summary_path)
            if isinstance(payload, dict):
                payload["capabilities"] = {
                    "browser_onnx": True,
                    "server_cpu_preview": True,
                    "gpu_benchmark": (ROOT / "scripts" / "benchmark_tracking.py").exists(),
                }
            self.write_json(payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/upload":
            self.handle_upload()
            return
        if self.path == "/api/run-benchmark":
            self.handle_benchmark()
            return
        if self.path == "/api/process-upload":
            self.handle_process_upload()
            return
        self.send_error(404, "unknown endpoint")

    def handle_upload(self) -> None:
        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            self.write_json({"error": "Expected multipart/form-data"}, status=400)
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        item = form["video"] if "video" in form else None
        if item is None or not getattr(item, "filename", ""):
            self.write_json({"error": "No video field in upload"}, status=400)
            return
        suffix = Path(item.filename).suffix.lower() or ".mp4"
        safe_name = f"upload_{int(time.time())}{suffix}"
        output_path = UPLOAD_DIR / safe_name
        with output_path.open("wb") as f:
            while True:
                chunk = item.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        rel_path = output_path.relative_to(ROOT).as_posix()
        self.write_json(
            {
                "path": rel_path,
                "command": benchmark_command(output_path),
            }
        )

    def handle_process_upload(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            video = resolve_video(payload.get("video"))
            output_dir = ROOT / "outputs" / "demo_site_uploads" / video.stem
            cmd = [
                sys.executable,
                "scripts/process_upload_demo.py",
                "--video",
                str(video),
                "--output-dir",
                str(output_dir),
                "--max-frames",
                str(int(payload.get("max_frames", 180))),
                "--max-side",
                "640",
            ]
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            metrics = read_json(output_dir / "metrics.json")
            output_video = Path(str(metrics["output_video"])).relative_to(ROOT).as_posix()
            self.write_json(
                {
                    "command": cmd,
                    "stdout": result.stdout[-4000:],
                    "metrics": metrics,
                    "output_video": f"/{output_video}",
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.write_json({"error": str(exc)}, status=500)

    def handle_benchmark(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            video = resolve_video(payload.get("video"))
            output_root = ROOT / "outputs" / "demo_site_live_benchmark"
            cmd = benchmark_command(video, output_root=output_root)
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            summary = read_json(output_root / "summary.json")
            self.write_json({"command": cmd, "stdout": result.stdout[-4000:], "results": summary})
        except Exception as exc:  # noqa: BLE001
            self.write_json({"error": str(exc)}, status=500)

    def write_json(self, payload: object, status: int = 200) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def guess_type(self, path: str) -> str:
        if path.endswith(".js"):
            return "application/javascript"
        if path.endswith(".css"):
            return "text/css"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_video(raw: str | None) -> Path:
    if not raw:
        return DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else PUBLIC_VIDEO
    requested = Path(raw)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("Video path must stay inside the repository")
    candidate = ROOT / requested
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def benchmark_command(video: Path, output_root: Path | None = None) -> list[str]:
    root = output_root or (ROOT / "outputs" / "demo_site_uploaded_benchmark")
    return [
        sys.executable,
        "scripts/benchmark_tracking.py",
        "--video",
        str(video),
        "--output-root",
        str(root),
        "--detector",
        "yolo",
        "--frames",
        "45",
        "--warmup-frames",
        "5",
        "--max-sides",
        "512",
        "--sam-every",
        "30",
        "--yolo-every",
        "10",
        "--yolo-model",
        "yolov8n.pt",
        "--yolo-imgsz",
        "480",
        "--no-sam-on-yolo",
        "--dtype",
        "bf16",
        "--device",
        "cuda",
    ]


def main() -> None:
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8765))
    host = "0.0.0.0" if os.environ.get("FLY_APP_NAME") else "127.0.0.1"
    httpd = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"serving http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
