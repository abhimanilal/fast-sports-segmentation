# Client-Side ML Roadmap

The public demo should not depend on pre-rendered mask videos. The current browser path proves the system boundary: frames stay local, YOLO-Seg ONNX runs in the browser, masks are decoded in JavaScript, IDs are associated over time, and overlays are painted on a canvas.

The remaining gap is mask quality. Generic YOLO11n-Seg at 320 input is fast enough to be interactive, but its masks are coarse on small basketball players and it misses/over-selects in crowded frames. The ML work is to build a sports-specific browser student rather than relying on the generic COCO segmentation head.

## Target

- Client-side mask tracking on uploaded or sample sports video.
- No replayed mask videos in the product demo.
- 8-12 processed mask frames per second on a modern laptop browser.
- Cleaner player masks than generic YOLO11n-Seg 320 at the same or lower runtime.
- Stable IDs through ordinary cuts, occlusions, and camera motion.

## Approach

1. Generate teacher masks offline.
   - Run the best local teacher pipeline on the basketball clips: YOLO/EdgeTAM/SAM-style propagation plus ROI filtering.
   - Store pseudo-labels as per-frame boxes, masks, track IDs, and confidence.
   - Keep hard negatives: sideline people, walls, scoreboard overlays, benches, and hoop structures.

2. Train a browser student.
   - Start from YOLO11n-Seg or a smaller segmentation head.
   - Fine-tune only on sports-person masks and hard negatives.
   - Optimize for player recall at small scale, not generic COCO segmentation.
   - Export ONNX at 320 input first; evaluate 256 only if quality holds.

3. Add temporal learning.
   - Distill teacher track consistency into the student with adjacent-frame pairs.
   - Add a lightweight mask-propagation/refinement module that takes previous mask, current crop, and current box.
   - Run heavy detection sparsely and cheap refinement/propagation on intervening frames.

4. Optimize the browser runtime.
   - Prefer WebGPU, fall back to WASM.
   - Quantize after quality is acceptable.
   - Keep masks crop-local where possible instead of decoding full-frame masks for every detection.
   - Move decode/association into a Web Worker or OffscreenCanvas path to protect UI responsiveness.

## Benchmarks

Track every candidate with both quality and browser runtime:

- Mean mask IoU against teacher pseudo-labels.
- Player recall on representative frames.
- False positives per frame.
- ID switches per 100 frames.
- Processed mask FPS in browser.
- p95 UI-frame blocking time.

The resume claim should be: `trained and deployed a browser-side sports-player mask tracker distilled from a stronger video teacher`, not just `ran an off-the-shelf segmentation model in the browser`.
