# Fast Sports Segmentation TODO

## Realtime Analytics Bar

For this to read as the base of a realtime sports analytics system, the demo should show three layers:

1. Realtime inner loop: ingest frames, maintain player tracks, periodically refresh masks, and expose per-player geometry at game speed.
2. Async detector loop: run heavier open-vocabulary detection only for initialization, recovery, scene changes, or new classes.
3. Analytics layer: convert masks/tracks into useful sports signals such as player occupancy, court zones, speed estimates, possession candidates, and event clips.

Minimum local benchmark target for an impressive portfolio demo:

- 30 FPS effective processing on 640-720p frames.
- p95 frame time under 33 ms when debug rendering and disk writes are disabled.
- SAM refresh under 100 ms p95 for 4-8 players, amortized with `--sam-every`.
- Detector/reseed path isolated from the frame loop. LocateAnything-3B is acceptable as an async open-vocabulary fallback, but not as the realtime detector.
- Stable recovery behavior: stale tracks disappear, and forced reseed happens when all tracks are pruned.

Stretch target:

- 60 FPS inner loop at 640p, or 30 FPS at 720p with overlays enabled.
- GPU memory below 6 GB for the realtime path.
- Multiple sports clips with a single command benchmark report.

## Current Architecture

- LocateAnything-3B: open-vocabulary box reseeding.
- EfficientSAM3-S vision-only checkpoint: box-prompt segmentation.
- Template matching: frame-to-frame box propagation.
- SAM score and box filters: admission/pruning.

## Immediate TODOs

- Add benchmark fixtures for all five informal basketball clips.
- Make detector accuracy a first-class metric for YOLOv8n vs YOLO26n selection: report precision/recall or mAP on labeled sports frames, plus missed-player and false-positive counts on the informal basketball clips.
- Pin the LocateAnything remote-code revision to avoid surprise Hugging Face code updates.
- Add CI-light tests for box filtering, NMS, and metrics parsing.
- Add multi-object identity quality metrics, not just speed metrics.
- Upgrade association from heuristic IoU/center matching to Kalman plus appearance embeddings.
- Add court-zone occupancy summaries on top of the current court-coordinate export.
- Add periodic EdgeTAM correction/reseed support, not only frame-0 YOLO seeds.
- Benchmark EdgeTAM with one to three objects separately from full-team tracking.
- Try TensorRT/ONNX exports for YOLOv8n and YOLO26n, but only replace the current detector after latency and detector-accuracy comparisons both pass.
- Expand SA-V benchmark from 3 videos to 25-50 videos and report confidence intervals.

## Done

- Created local project repo with ignored data/checkpoints/vendor/output folders.
- Slimmed EfficientSAM3 to vision/interactivity weights for a 143.7 MiB checkpoint.
- Added LocateAnything -> EfficientSAM sparse tracking path.
- Added YouTube Creative Commons informal basketball clips and short FFmpeg excerpts.
- Added detector box post-processing: shape filters, edge rejection, IoU/containment NMS.
- Added SAM-score track admission and pruning.
- Added forced detector reseed when all tracks are pruned.
- Added benchmark mode flags for no frame writes, no overlays, and per-frame timing.
- Added YOLOv8n realtime person detector path.
- Kept LocateAnything as async semantic reseeding rather than the realtime detector.
- Hit the realtime benchmark target on the rec-league clip with YOLO every 10 frames, SAM every 30 frames, 512 detector input, BF16: 98.55 FPS post-warmup, p95 32.14 ms, 100% active frames.
- Added persistent track IDs across detector reseeds with IoU/center-distance association and a short miss buffer.
- Added analytics export for persistent IDs, image-space speed, optional court-plane coordinates, and optional court-plane speed.
- Added an approximate court homography fixture for the rec-league basketball clip.
- Hit the realtime benchmark target with persistent IDs using YOLO every 10 frames, SAM every 30 frames, 480 detector input, BF16: 96.13 FPS post-warmup, p95 25.73 ms, 100% active frames.
- Validated YOLO26n as a drop-in Ultralytics detector, but it missed the p95 realtime target locally at 480 and 416 detector inputs.
- Added an experimental YOLO-seeded EdgeTAM video-memory runner and validated a 60-frame BF16 smoke run at 9.18 FPS propagation.
- Added SA-V Subset 51 shard export and EdgeTAM oracle-box accuracy benchmark.
- Measured EdgeTAM on 3 SA-V Subset 51 videos: 617 object-frame pairs, 0.816 mean mask IoU, 0.911 J@0.5, 10.86 FPS propagation.
- Added EfficientSAM sparse-refresh SA-V benchmark and comparison report.
- Measured EfficientSAM on the same 3-video SA-V shard: SAM every 5 frames reached 32.91 FPS with 0.211 refresh-frame mask IoU; SAM every 30 frames reached 48.98 FPS with 0.401 refresh-frame mask IoU.
- Added a portfolio report generator and tracked `PORTFOLIO.md` with resume positioning, sports realtime metrics, seed-detector tradeoffs, and SA-V accuracy comparison.
- Added a polished side-by-side portfolio demo exporter and validated a 60-frame 2340x540 demo clip.
- Added start-frame aligned YOLO-seeded EdgeTAM chunks and validated a 450-frame, 15.015-second, 2340x540 portfolio demo clip.
