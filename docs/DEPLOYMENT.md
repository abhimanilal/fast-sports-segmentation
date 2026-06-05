# Deployable Demo

This repo has two demo modes:

1. Public web demo for Fly.io: upload a clip, run YOLOv8n ONNX directly in the browser with ONNXRuntime Web, and optionally generate a CPU-safe server preview overlay.
2. Local research demo: run the full CUDA benchmark and EdgeTAM/SAM artifacts from the project venv.

## Fly.io

The public deployment uses `Dockerfile`, `requirements-demo.txt`, and `fly.toml`. It avoids the CUDA research stack so the app can run on a small shared CPU machine.

```powershell
fly launch --no-deploy
fly deploy
```

If Fly asks for an app name, use `fast-sports-segmentation` or another available slug. The app is served on port `8080`.

## Browser ONNX

The browser detector uses:

- `media/models/yolov8n_480_nms.onnx`
- ONNXRuntime Web from CDN
- uploaded or bundled video frames sampled in the browser
- person-class detections drawn on a canvas

This is intentionally a detector overlay, not the full EdgeTAM mask stack. It gives recruiters a live, no-GPU interaction path on an uploaded clip while the repo documents the heavier local mask-propagation benchmark.

## Transformers.js

Transformers.js is a reasonable next packaging layer because it runs ONNX models in the browser through ONNX Runtime and supports computer-vision tasks. For this repo, direct ONNXRuntime Web is the better first integration because the Ultralytics export already emits a compact ONNX model with NMS output.

The next Transformers.js step would be to publish a browser-ready model repo on Hugging Face with:

- `onnx/model.onnx`
- model config and preprocessor metadata
- a small wrapper that maps video frames to YOLO input and draws detections

## Local GPU Mode

Run the local site:

```powershell
.venv\Scripts\python.exe scripts\serve_demo.py 8765
```

Open:

```text
http://127.0.0.1:8765
```

The local site enables the CUDA benchmark endpoint when `scripts/benchmark_tracking.py` is present.
