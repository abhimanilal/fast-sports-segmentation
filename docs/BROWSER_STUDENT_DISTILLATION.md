# Browser Student Distillation

## Literature Read

- SAM 2 frames video segmentation around streaming memory: a prompt initializes an object and memory carries it through later frames. That is the accuracy model this project benchmarks against, but the full stack is too heavy for the browser demo target.
- EdgeTAM and EfficientTAM compress the SAM 2-style track-anything path for on-device use, which supports the repo's offline teacher/benchmark direction.
- Instance-segmentation distillation work supports the teacher/student setup used here: run a stronger teacher offline, filter pseudo-labels, then train a smaller student optimized for the deployment constraint.
- ONNX Runtime Web with WebGPU/WASM is the deployable browser runtime here. The product path must therefore keep model size small, avoid server-side mask rendering, and decode overlays locally.

Primary sources used:

- SAM 2: https://arxiv.org/abs/2408.00714
- EdgeTAM: https://huggingface.co/papers/2501.07256
- EfficientTAM: https://huggingface.co/papers/2411.18933
- ONNX Runtime WebGPU docs: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- Hierarchical distillation for efficient instance segmentation: https://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Deng_Towards_Efficient_Instance_Segmentation_with_Hierarchical_Distillation_ICCVW_2019_paper.pdf

## Student Setup

The browser model is a one-class YOLO11n-Seg student trained for `player`, exported as static ONNX at `320 x 320` input:

- Teacher: `yolo11s-seg.pt`
- Student init: `yolo11n-seg.pt`
- Initial training data: 434 frames from five local basketball clips
- Initial pseudo-labels: 2,752 teacher masks
- Active-play cleanup: 548 teacher person detections rejected by normalized ROI foot-point filtering
- Auxiliary GT: 346 sampled frames from a 19-video SA-V sports/action shard
- Current export: `media/models/yolo11n_sports_roi_sav19_student_320.onnx`
- Previous export: `media/models/yolo11n_sports_student_320.onnx`

The first SA-V tiny-shard fine-tune was rejected. It validated poorly and visually stopped detecting basketball players, which is expected because the shard was generic object-video segmentation rather than target-domain sports players.

## Benchmark

All rows below validate on the same 88-frame ROI-filtered teacher-label validation set.

| Model | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box mAP50-95 | GPU inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic YOLO11n-Seg 320 | 0.687 | 0.338 | 0.728 | 0.411 | 3.7 ms |
| Earlier non-ROI sports student | 0.589 | 0.294 | 0.642 | 0.355 | 4.5 ms |
| ROI-distilled sports student | 0.732 | 0.394 | 0.773 | 0.482 | 4.2 ms |
| ROI + balanced SA-V GT student | 0.765 | 0.406 | 0.810 | 0.516 | 2.6 ms |

The balanced mixed student is the current browser candidate because it improves the sports-domain validation target without increasing model size.

## SA-V Sports/Action Shard

A broad SA-V tiny-shard fine-tune was not representative enough for basketball. To avoid hand labels, the repo includes a semantic selector that searches SA-V Subset 51 for court/sports-like videos:

```powershell
.venv\Scripts\python.exe scripts\select_sav_sports_shard.py --max-candidates 917 --top-k 64 --frames-per-video 5 --output-dir data\raw\sav_sports_candidates_full
.venv\Scripts\python.exe scripts\export_sav_selected_shard.py --selected-ids examples\sav_sports_court_selected_ids.txt --output-dir data\raw\sav_sports_action_shard19 --annotation-field manual --overwrite
.venv\Scripts\python.exe scripts\export_sav_yolo_seg_dataset.py --manifest data\raw\sav_sports_action_shard19\manifest.json --output-dir data\derived\sav_sports_action19_yolo_seg --max-samples 19 --max-frames 120 --max-side 640 --val-samples 4 --min-area-frac 0.001 --max-objects-per-frame 12
```

The expanded selected shard contains 19 visually relevant sports/action videos, 1,571 annotated frames, and 5,542 YOLO-format mask instances. The labels are real SA-V manual masklets, not YOLO pseudo-labels.

Validation on the 317-frame SA-V sports/action holdout:

| Model | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Generic YOLO11n-Seg 320 | 0.273 | 0.150 | 0.363 | 0.260 |
| ROI-distilled sports student | 0.236 | 0.091 | 0.279 | 0.139 |
| SA-V-only GT student | 0.182 | 0.071 | 0.329 | 0.171 |
| ROI then SA-V GT, low LR | 0.228 | 0.097 | 0.340 | 0.192 |
| ROI then SA-V GT, frozen backbone | 0.267 | 0.090 | 0.378 | 0.191 |
| ROI + balanced SA-V GT student | 0.222 | 0.085 | 0.366 | 0.212 |

The expanded shard changed the conclusion from the tiny 4-video test. Generic YOLO is still the strongest SA-V GT baseline, while direct SA-V fine-tuning damages basketball-domain recall. The frozen SA-V model is closest on SA-V but visually misses too many active players. The deployed path is therefore mixed training: use SA-V as auxiliary mask supervision, select on sports-domain validation, and keep SA-V as an external accuracy audit.

## Visual Review

Accepted improvements:

- Better recall on small and distant players in the rec-league and pickup clips.
- Cleaner behavior than the first target-domain student, which over-learned broad person labels.
- The mixed student recovers more active players in crowded rec-league frames than the previous ROI-only export.
- Masks are good enough for a live browser overlay at 320 input when paired with ROI postprocessing.

Remaining issues:

- Rec-league baseline-area players and nearby sideline people can still be ambiguous.
- POV clips can include the camera wearer or close foreground bodies as valid masks.
- ID stability still comes from JavaScript association, not a learned temporal memory module.

## Next ML Iteration

- Add manually reviewed hard negatives for bench/sideline people.
- Train with adjacent-frame consistency so the student learns track stability, not only per-frame masks.
- Move browser decode and association into a worker/offscreen canvas path before increasing input size.
- Add a player-only SA-V/person-overlap filter so SA-V object masklets do not reward non-player objects.
