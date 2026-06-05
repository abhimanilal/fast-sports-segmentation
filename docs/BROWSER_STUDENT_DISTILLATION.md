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
- Training data: 434 frames from five local basketball clips
- Pseudo-labels: 2,752 teacher masks
- Active-play cleanup: 548 teacher person detections rejected by normalized ROI foot-point filtering
- Export: `media/models/yolo11n_sports_student_320.onnx`

The first SA-V tiny-shard fine-tune was rejected. It validated poorly and visually stopped detecting basketball players, which is expected because the shard was generic object-video segmentation rather than target-domain sports players.

## Benchmark

All rows below validate on the same 88-frame ROI-filtered teacher-label validation set.

| Model | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box mAP50-95 | GPU inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic YOLO11n-Seg 320 | 0.687 | 0.338 | 0.728 | 0.411 | 3.7 ms |
| Earlier non-ROI sports student | 0.589 | 0.294 | 0.642 | 0.355 | 4.5 ms |
| ROI-distilled sports student | 0.732 | 0.394 | 0.773 | 0.482 | 4.2 ms |

The ROI-distilled student is the only candidate that improves over the generic browser model on both mask agreement and box agreement.

## Visual Review

Accepted improvements:

- Better recall on small and distant players in the rec-league and pickup clips.
- Cleaner behavior than the first target-domain student, which over-learned broad person labels.
- Masks are good enough for a live browser overlay at 320 input when paired with ROI postprocessing.

Remaining issues:

- Rec-league baseline-area players and nearby sideline people can still be ambiguous.
- POV clips can include the camera wearer or close foreground bodies as valid masks.
- ID stability still comes from JavaScript association, not a learned temporal memory module.

## Next ML Iteration

- Add manually reviewed hard negatives for bench/sideline people.
- Train with adjacent-frame consistency so the student learns track stability, not only per-frame masks.
- Move browser decode and association into a worker/offscreen canvas path before increasing input size.
- Add a small hand-labeled basketball validation set for true player precision/recall; teacher-label mAP is useful but not a final accuracy claim.
