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

- Add a stable track identity layer with Kalman prediction and association by IoU/appearance.
- Add benchmark fixtures for all five informal basketball clips.
- Add analytics outputs: player center, mask area, court-zone occupancy, and simple speed estimates.
- Add a `scripts/export_demo_clip.py` command for polished before/after demo videos.
- Pin the LocateAnything remote-code revision to avoid surprise Hugging Face code updates.
- Add CI-light tests for box filtering, NMS, and metrics parsing.
- Add multi-object identity quality metrics, not just speed metrics.
- Add a lightweight court calibration step so tracks become court-space analytics, not only image-space overlays.

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
