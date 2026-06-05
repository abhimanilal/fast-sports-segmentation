# Fast Sports Segmentation Portfolio Report

## Resume Positioning

Built a realtime sports-video segmentation and analytics pipeline using YOLO/LocateAnything seeding, EdgeTAM video mask propagation, and court-space tracking analytics; benchmarked against EfficientSAM3 on SA-V with mask IoU, J@0.5, and FPS tradeoff analysis.

## What The System Does

- Detects player seed boxes with YOLO by default, with LocateAnything available for slower open-vocabulary recovery.
- Propagates masks through video with EdgeTAM.
- Exports per-frame tracking geometry and approximate court-space analytics.
- Benchmarks segmentation quality on SA-V and speed on local sports clips.

## Sports Realtime Profile

Measured on `rec_league_0008_45s.mp4` with YOLOv8n, persistent IDs, BF16, 512-side frames, YOLO every 10 frames, and SAM refresh every 30 frames:

- FPS after warmup: `96.13`
- p50 frame time: `6.24 ms`
- p95 frame time: `25.77 ms`
- Active-frame coverage: `100.0%`
- Average tracks: `8.07`

Analytics export:

- Frames: `150`
- Analytics rows: `1210`
- Track IDs: `14`
- Tracks lasting at least 60 frames: `10`
- Court projection units: `meters`

## Polished Demo Artifact

- Video: `outputs\portfolio_demo_15s\sports_segmentation_demo_15s.mp4`
- GitHub copy: `media\sports_segmentation_demo_15s.mp4`
- Frames: `450`
- Resolution: `2340x540`

## Seed Detector Comparison

Measured on the same 60-frame rec-league EdgeTAM run:

| Seed Detector | Seed Time | Raw Boxes | Filtered | Kept | EdgeTAM FPS | End-to-End |
| --- | --- | --- | --- | --- | --- | --- |
| yolo | 2.31s | 8 | 2 | 6 | 7.97 | 14.61s |
| locate | 87.21s | 170 | 162 | 6 | 8.81 | 98.69s |

YOLO is the correct default for realtime player seeding. LocateAnything is useful as an async semantic recovery path, not as a live seed detector.

## SA-V Accuracy Comparison

All rows use the same 3-video SA-V Subset 51 shard with oracle first-frame boxes.

| System | Cadence | Mask Pairs | Mean Mask IoU | Mask J@0.5 | Mean Box IoU | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| EdgeTAM oracle boxes | video memory every frame | 617 | 0.816 | 0.911 | 0.803 | 10.86 |
| EfficientSAM sparse refresh 5 | SAM every 5 frames | 124 | 0.211 | 0.185 | 0.282 | 32.91 |
| EfficientSAM sparse refresh 30 | SAM every 30 frames | 22 | 0.401 | 0.455 | 0.301 | 48.98 |

Interpretation: EdgeTAM is the accuracy winner by a wide margin. EfficientSAM3 is faster in the sparse template-tracking setup, but without video memory it is not competitive on mask tracking quality.

## Portfolio Claims Supported By Metrics

- Built and benchmarked an end-to-end video segmentation system rather than only running a model.
- Chose EdgeTAM over EfficientSAM3 for mask propagation based on measured SA-V accuracy.
- Quantified detector tradeoffs: YOLO seed latency is seconds, LocateAnything seed latency is tens of seconds on this host.
- Added court-space analytics and persistent IDs to connect segmentation output to sports analytics.

## Next Work

- Add ByteTrack/DeepSORT-grade identity association for crowded sports footage.
- Add periodic EdgeTAM correction prompts from YOLO when players enter or occlude.
- Expand SA-V evaluation from 3 videos to 25-50 videos with confidence intervals.
