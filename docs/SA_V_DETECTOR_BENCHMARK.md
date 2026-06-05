# SA-V Detector Benchmark

This benchmark compares YOLOv8n and YOLO26n as class-agnostic seed-box detectors on the same 3-video SA-V Subset 51 shard used for the EdgeTAM/EfficientSAM mask benchmarks.

Important caveat: SA-V is a class-agnostic video object segmentation dataset, not a sports-player detection dataset. These results measure whether a detector can cover SA-V masklet boxes well enough to seed video tracking. They do not replace a labeled basketball-player benchmark.

Command:

```powershell
.venv\Scripts\python.exe scripts\benchmark_sav_yolo_detectors.py `
  --manifest data\raw\sav_subset51_shard_3\manifest.json `
  --output-dir outputs\sav_yolo_detector_comparison `
  --models yolov8n.pt yolo26n.pt `
  --max-samples 3 `
  --max-frames 60 `
  --max-side 512 `
  --imgsz 480 `
  --conf 0.001 `
  --operating-conf 0.25 `
  --device 0
```

Metrics:

- `AP50` / `AP75`: area under the precision-recall curve using low-confidence detections.
- `P/R/F1 @ 0.25`: operating-point precision, recall, and F1 at confidence `0.25`.
- Matching is class-agnostic against SA-V manual masklet boxes.
- Duplicate ground-truth boxes are removed at IoU `>= 0.95`.

Results:

| Model | Frames | GT Boxes | Pred @ 0.25 | Raw Pred | FPS | p50 ms | p95 ms | AP50 | AP75 | P@50 | R@50 | F1@50 | P@75 | R@75 | F1@75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 180 | 587 | 282 | 13,945 | 48.17 | 13.13 | 24.71 | 0.353 | 0.325 | 0.652 | 0.313 | 0.423 | 0.638 | 0.307 | 0.414 |
| YOLO26n | 180 | 587 | 292 | 14,113 | 47.15 | 19.12 | 26.34 | 0.342 | 0.321 | 0.630 | 0.313 | 0.419 | 0.616 | 0.307 | 0.410 |

Interpretation:

YOLOv8n remains the better default for the deployable browser detector on this evidence. YOLO26n does not improve SA-V seed-box AP or recall on this shard and is slower at the measured `480` input profile.

Next detector-quality step:

Create a small labeled sports-player validation set from the informal basketball clips and report player-specific mAP, missed-player counts, false-positive counts, and downstream ID/track coverage.
