# Fast Sports Segmentation

Smoke-test repo for fast sports-video segmentation with EfficientSAM3 model-zoo checkpoints.

## Demo Asset

The starter demo uses Wikimedia Commons `Gant Windup.webm`, a 14 second 1080p clip of John Gant pitching for Atlanta on June 17, 2016.

Professional sports video sources worth using next:

- SoccerNet: best research-grade source for full broadcast soccer games, but video access requires signing its NDA.
- SportsMOT / MultiSports-style datasets: good benchmark direction for tracking and sports scenes, but verify download terms before redistribution.
- Wikimedia Commons sports clips: easiest for public demos because files and licensing metadata are accessible.

Download local sports test clips:

```powershell
.venv\Scripts\python.exe scripts\download_sample_videos.py
```

Current local samples:

- `data\raw\basketball_wheelchair.webm`
- `data\raw\american_football_kickoff.webm`
- `data\raw\soccer_beautiful_game.webm`

I avoided downloading NBA/NFL/soccer highlight uploads from YouTube directly because those are typically copyrighted uploads. The downloader uses Wikimedia Commons clips and writes source/license notes to `data\raw\sample_videos_manifest.json`.

## Informal Basketball Test Clips

For more representative basketball tracking tests, this repo also has YouTube clips selected from Creative Commons-filtered search results. They are downloaded as progressive 360p MP4 files so OpenCV can read them without requiring `ffmpeg` to merge separate video/audio streams.

Local files:

- `data\raw\youtube\PVSCNealbRI_LA_Fitness_Open_Gym_1_24_18_G2.mp4`
- `data\raw\youtube\6ajFxyTZkNw_Old_Man_COOKS_Teenagers_in_5v5_Pickup_Basketball_Game.mp4`
- `data\raw\youtube\8qiOAz4CpPc_Pickup_Ball_Nov_6_2013.mp4`
- `data\raw\youtube\-IHn5OJFFSM_2026_Richmond_Spring_Rec_League_-_Cream_City_Snipers_vs._Meat_Snackers_-_Roundba.mp4`
- `data\raw\youtube\_ZV2rqcZJWc_Street_basketball_first_person_POV_episode_1_live_a_lot_of_sports.mp4`

Metadata and license strings from `yt-dlp` are in `data\raw\youtube\youtube_basketball_manifest.json`, with the original per-video `.info.json` files next to the videos. A quick visual preview is at `outputs\previews\youtube_basketball_contact_sheet.jpg`.

Short 45 second excerpts for faster iteration:

- `data\raw\youtube\clips\rec_league_0008_45s.mp4`
- `data\raw\youtube\clips\la_fitness_open_gym_0060_45s.mp4`
- `data\raw\youtube\clips\pickup_ball_0020_45s.mp4`
- `data\raw\youtube\clips\pickup_5v5_0090_45s.mp4`
- `data\raw\youtube\clips\streetball_pov_0010_45s.mp4`

FFmpeg is installed on this host via `winget install --id Gyan.FFmpeg --exact`. The excerpt command pattern is:

```powershell
ffmpeg -ss 00:00:08 -i input.mp4 -t 45 -vf "scale='min(640,iw)':-2" -an -c:v libx264 -preset veryfast -crf 23 -movflags +faststart output.mp4
```

Validated locally with OpenCV on June 5, 2026. A sparse LocateAnything plus EfficientSAM tracking run on the rec-league clip produced:

- `outputs\track_youtube_rec_league_sparse\tracked.mp4`
- `outputs\track_youtube_rec_league_sparse\frame_00240.jpg`
- `outputs\track_youtube_rec_league_sparse\frame_00280.jpg`

LocateAnything output is post-processed before tracks are admitted:

- box shape and area filters remove tiny, huge, flat, and broad boxes
- frame-edge rejection removes cropped player/person boxes
- IoU plus containment NMS removes duplicate and nested detections
- detector reseeds always run through EfficientSAM before display
- SAM-score pruning removes stale tracks on every mask refresh
- if pruning removes every track, the loop forces a fresh LocateAnything reseed instead of failing

Validated run:

```powershell
.venv\Scripts\python.exe scripts\track_segment_video.py `
  --video data\raw\youtube\clips\rec_league_0008_45s.mp4 `
  --sam-checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --output-dir outputs\track_youtube_rec_league_postproc_final `
  --categories "basketball player" `
  --locate-every 45 `
  --sam-every 5 `
  --max-frames 75 `
  --max-side 512 `
  --device cuda `
  --dtype bf16 `
  --locate-device cuda `
  --locate-dtype bf16 `
  --locate-load-in-8bit `
  --locate-max-gpu-memory 5200MiB `
  --locate-max-cpu-memory 8GiB `
  --locate-offload-folder E:/HFOffload/LocateAnything8bit `
  --max-detections 8 `
  --write-video
```

Output:

- `outputs\track_youtube_rec_league_postproc_final\tracked.mp4`
- `outputs\track_youtube_rec_league_postproc_final\metrics.json`
- `outputs\track_youtube_rec_league_postproc_final\frame_00045.jpg`
- `outputs\track_youtube_rec_league_postproc_final\frame_00070.jpg`
- `outputs\track_youtube_rec_league_postproc_final\frame_00071.jpg`

On this host, LocateAnything still takes about 67-69 seconds per detector call in 8-bit BF16 mode. SAM refreshes are fast, around 0.06-0.36 seconds in this run. Remaining quality issue: the broad prompt `basketball player` can still admit semantic false positives such as sideline/referee people; the next refinement should use a narrower prompt or add court/jersey priors.

## Setup

This repo vendors EfficientSAM3 under `vendor/efficientsam3`.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -e vendor\efficientsam3 opencv-python pillow matplotlib scipy scikit-image einops pycocotools omegaconf
```

On this machine, `.venv`, `data`, `models`, `outputs`, and `vendor` are junctioned to `E:` because `C:` was nearly full.

Optional LocateAnything detector dependencies:

```powershell
uv pip install --python .venv\Scripts\python.exe transformers==4.57.1 peft decord lmdb accelerate sentencepiece protobuf bitsandbytes
```

Optional realtime detector dependency:

```powershell
uv pip install --python .venv\Scripts\python.exe ultralytics
```

## Run

```powershell
.venv\Scripts\python.exe scripts\segment_video_point_prompt.py `
  --video data\raw\gant_windup.webm `
  --checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --output-dir outputs\gant_efficientvit_s `
  --point-x-frac 0.492 `
  --point-y-frac 0.509 `
  --stride 48 `
  --max-side 1024 `
  --dtype fp16
```

The current EfficientSAM3 model-zoo release has Stage 1 image/text weights available. Stage 2 video-memory and Stage 3 end-to-end video tracking weights are still marked planned upstream, so this runner samples video frames and segments with image prompts.

## LocateAnything -> EfficientSAM3

The two-stage pipeline is:

1. LocateAnything-3B detects/open-vocabulary grounds boxes from text prompts.
2. Vision-only EfficientSAM3-S segments inside those boxes with `predict_inst(..., box=...)`.

LocateAnything-3B is a 3B VLM from NVIDIA, released May 26, 2026, under NVIDIA's non-commercial research license. Its model card recommends Transformers inference, BF16, and Linux/NVIDIA GPUs; it is not a small real-time detector for an 8 GB card. In this repo it should be used sparsely, for example every N frames, with masks propagated/tracked between detections.

Run the integrated script with precomputed boxes:

```powershell
.venv\Scripts\python.exe scripts\detect_then_segment_video.py `
  --video data\raw\gant_windup.webm `
  --sam-checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --boxes-json examples\gant_sample_boxes.json `
  --output-dir outputs\gant_boxes_to_sam_smoke `
  --stride 60 `
  --max-frames 2 `
  --max-side 512 `
  --sam-device cuda `
  --sam-dtype fp16
```

Run with LocateAnything detection:

```powershell
$env:HF_HOME="E:\HuggingFace"
$env:TRANSFORMERS_CACHE="E:\HuggingFace\transformers"
.venv\Scripts\python.exe scripts\detect_then_segment_video.py `
  --video data\raw\gant_windup.webm `
  --sam-checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --categories "baseball player" `
  --output-dir outputs\gant_locate_anything_to_sam `
  --stride 60 `
  --max-frames 2 `
  --max-side 512 `
  --generation-mode fast `
  --detect-every 1 `
  --sam-dtype fp16 `
  --locate-dtype fp16
```

The current EfficientSAM3 builder in `vendor/efficientsam3` has a local `enable_text_encoder=False` patch so the SAM stage instantiates only the vision/interactive-prompt path. That keeps the SAM checkpoint at 143.7 MiB and avoids constructing the unused EfficientSAM language encoder.

Both stages expose dtype switches. The default CUDA path is FP16:

- EfficientSAM3: `--sam-dtype fp16` or `--dtype fp16`
- LocateAnything: `--locate-dtype fp16`

FP16 reduces the live model memory footprint versus FP32. BF16 is also validated with `bf16`; it works well for the current LocateAnything/SAM path.

## Live Tracking Loop

The tracking loop supports three sources of boxes:

- `--boxes-json`: seed or correction boxes from a file.
- `--yolo-every N`: run a lightweight YOLO person detector every `N` processed frames.
- `--locate-every N`: run LocateAnything every `N` processed frames.
- template matching: lightweight frame-to-frame box propagation between detector calls.

The intended live loop is:

```text
YOLO realtime boxes -> template tracking -> periodic EfficientSAM3 masks
LocateAnything sparse semantic reseed -> EfficientSAM3 box masks -> template tracking until next detection
```

Run with seed boxes only:

```powershell
.venv\Scripts\python.exe scripts\track_segment_video.py `
  --video data\raw\basketball_wheelchair.webm `
  --sam-checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --boxes-json examples\basketball_wheelchair_seed_boxes.json `
  --output-dir outputs\track_basketball_wheelchair_smoke `
  --max-frames 45 `
  --max-side 512 `
  --sam-every 10 `
  --device cuda `
  --dtype fp16 `
  --write-video
```

Run with LocateAnything inside the loop:

```powershell
$env:HF_HOME="E:\HuggingFace"
$env:TRANSFORMERS_CACHE="E:\HuggingFace\transformers"
.venv\Scripts\python.exe scripts\track_segment_video.py `
  --video data\raw\basketball_wheelchair.webm `
  --sam-checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --output-dir outputs\track_basketball_locate_loop `
  --categories "basketball player" `
  --locate-every 30 `
  --sam-every 5 `
  --max-frames 120 `
  --max-side 512 `
  --device cuda `
  --dtype bf16 `
  --locate-device cuda `
  --locate-dtype bf16 `
  --locate-load-in-8bit `
  --locate-max-gpu-memory 5200MiB `
  --locate-max-cpu-memory 8GiB `
  --locate-offload-folder E:/HFOffload/LocateAnything8bit `
  --write-video
```

On an 8 GB RTX 3070, start with `--locate-every 30` or higher. If both models do not fit together, keep SAM on CUDA and run LocateAnything less frequently, or move the LocateAnything call to a separate process/service later.

Run the realtime-oriented YOLO path:

```powershell
.venv\Scripts\python.exe scripts\track_segment_video.py `
  --video data\raw\youtube\clips\rec_league_0008_45s.mp4 `
  --sam-checkpoint models\efficient_sam3_efficientvit_s_point_prompt_slim.pt `
  --output-dir outputs\track_youtube_rec_league_yolo_smoke `
  --yolo-every 10 `
  --yolo-model yolov8n.pt `
  --yolo-conf 0.25 `
  --yolo-imgsz 512 `
  --no-sam-on-yolo `
  --sam-every 30 `
  --max-frames 150 `
  --max-side 512 `
  --device cuda `
  --dtype bf16 `
  --max-detections 8 `
  --write-video
```

This keeps YOLO as the cheap realtime detector and lets SAM refresh masks on its own cadence. Use `--sam-on-yolo` when you want every detector reseed validated by SAM, but that is slower and misses the realtime frame budget on this host.

## Benchmarks

Realtime target for the inner loop:

- 30 FPS or better after warmup
- p95 frame time under 33 ms
- 100% active-frame coverage on the benchmark clip
- detector/reseed path kept separate from the expensive LocateAnything open-vocabulary path

Benchmark command:

```powershell
.venv\Scripts\python.exe scripts\benchmark_tracking.py `
  --detector yolo `
  --frames 150 `
  --warmup-frames 10 `
  --max-sides 512 `
  --sam-every 30 `
  --yolo-every 10 `
  --yolo-imgsz 512 `
  --no-sam-on-yolo `
  --dtype bf16 `
  --device cuda `
  --output-root outputs\benchmarks_realtime_yolo512
```

Validated on this host with `rec_league_0008_45s.mp4`:

| Profile | FPS | p50 frame | p95 frame | Active frames | Avg tracks | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLOv8n every 10, SAM every 30, 512 input, BF16 | 98.55 | 5.53 ms | 32.14 ms | 100% | 6.36 | Passes 30 FPS p95 target |

SAM remains the expensive stage at about 69 ms average for refresh frames, so this is not a synchronous per-frame mask system. The viable realtime design is frequent cheap box detection/tracking plus sparse mask refresh, with LocateAnything reserved for async semantic recovery.

## Analytics Export

Export first-pass image-space analytics from a tracking metrics file:

```powershell
.venv\Scripts\python.exe scripts\export_tracking_analytics.py `
  --metrics outputs\benchmarks_realtime_yolo512\side512_sam30\metrics.json `
  --output-csv outputs\benchmarks_realtime_yolo512\side512_sam30\analytics.csv `
  --summary-json outputs\benchmarks_realtime_yolo512\side512_sam30\analytics_summary.json
```

The CSV includes frame time, local track id, box center, box size, SAM score when available, detector source, and image-space speed. Speeds reset on detector reseed frames because the current tracker does not yet preserve identity across detections. That is the next required step before claiming player-level speed analytics.

Current LocateAnything local status on this Windows host:

- The project venv exists at `.venv`, junctioned to `E:\CodexVenvs\fast-sports-segmentation`.
- LocateAnything runtime dependencies are installed in that venv.
- Hugging Face cache is on `E:\HuggingFace`.
- LocateAnything-3B shards downloaded under `E:\HuggingFace\transformers`.
- After closing browser/Steam helper processes, LocateAnything loads in 8-bit and the sparse live loop runs.
- A local dtype bridge patch casts LocateAnything visual features to the language embedding dtype before insertion; this fixes the BF16/FP16 mismatch seen with 8-bit loading.
- A BF16 sparse run on `basketball_wheelchair.webm` succeeded with `--locate-every 30`, `--sam-every 5`, and `--max-detections 6`.
- LocateAnything first-frame detection took about 35 seconds on this host; SAM refreshes took about 0.09-0.35 seconds with six boxes in the short run.

The SAM side is using the smallest model-zoo family for this project, `ES-EV-S` / EfficientViT-B0, stripped to a 143.7 MiB vision-only point/box prompt checkpoint. The full EfficientSAM language encoder is not instantiated in the SAM runners.

## Slim Checkpoint

The raw EfficientSAM3 model-zoo checkpoints are large because they package more than the tiny EfficientViT image backbone. On this checkpoint:

- `efficient_sam3_efficientvit_s.pt`: 1.58 GiB file; about 1.35 GiB is `detector.backbone.language_backbone`.
- `efficient_sam3_efficientvit-b0_mobileclip_s1.pth`: 2.16 GiB file; about 1.81 GiB is `detector.backbone.vision_backbone`.

For point-prompt smoke tests, run:

```powershell
.venv\Scripts\python.exe scripts\make_point_prompt_checkpoint.py `
  --input models\efficient_sam3_efficientvit_s.pt `
  --output models\efficient_sam3_efficientvit_s_point_prompt_slim.pt
```

That produced `efficient_sam3_efficientvit_s_point_prompt_slim.pt`, a 143.7 MiB checkpoint containing the vision and interactive prompt/tracker weights.

## Current Smoke-Test Status

CUDA inference works with the slim checkpoint. On the RTX 3070 at `--max-side 512`, a four-frame sampled smoke test ran at about 0.058 seconds per frame after warmup, roughly 17.3 FPS. The fixed prompt point is good enough to prove the model path, but it over-segments later in the pitch motion; the next real project step is to drive the prompt from a detector/tracker instead of a fixed coordinate.

The box-prompt path is now validated with `examples/gant_sample_boxes.json`: after warmup, FP16 SAM box segmentation ran in about 0.066 seconds on the RTX 3070 and produced a clean pitcher mask. FP16 point-prompt segmentation ran at about 0.058 seconds per sampled frame after warmup, roughly 17.3 FPS.
