const state = {
  summary: null,
  uploadedPath: null,
  sourceVideoUrl: "/media/sports_segmentation_demo_15s.mp4",
  yoloSession: null,
  browserRunning: false,
};

const byId = (id) => document.getElementById(id);

function fmt(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function setText(id, value) {
  byId(id).textContent = value;
}

function renderSummary(summary) {
  state.summary = summary;
  const realtime = summary.sports_realtime || {};
  const analytics = summary.analytics || {};
  const sav = summary.sav_comparison || [];
  setText("fps", fmt(realtime.fps));
  setText("p95", fmt(realtime.p95_ms));
  setText("tracks", String(analytics.track_count || 0));
  const edge = sav.find((row) => row.name && row.name.startsWith("EdgeTAM")) || sav[0] || {};
  setText("iou", fmt(edge.mean_mask_iou, 3));
  const gpuButton = byId("run-benchmark");
  if (summary.capabilities && !summary.capabilities.gpu_benchmark) {
    gpuButton.disabled = true;
    gpuButton.textContent = "GPU benchmark is local-only";
  }

  const bars = byId("accuracy-bars");
  bars.innerHTML = "";
  sav.forEach((row, index) => {
    const value = Number(row.mean_mask_iou || 0);
    const item = document.createElement("div");
    item.className = "bar-row";
    item.innerHTML = `
      <div class="bar-label">
        <span>${row.name}</span>
        <strong>${value.toFixed(3)} IoU</strong>
      </div>
      <div class="bar-track"><div class="bar-fill ${index === 0 ? "" : "secondary"}"></div></div>
    `;
    bars.appendChild(item);
    requestAnimationFrame(() => {
      item.querySelector(".bar-fill").style.width = `${Math.min(100, value * 100)}%`;
    });
  });
}

async function loadSummary() {
  const response = await fetch("/api/summary");
  if (!response.ok) throw new Error(`summary failed: ${response.status}`);
  renderSummary(await response.json());
}

async function runBenchmark() {
  const button = byId("run-benchmark");
  const log = byId("benchmark-log");
  if (button.disabled) {
    log.textContent = "The deployed site runs browser ONNX and CPU upload previews. The full YOLO/SAM CUDA benchmark runs from the local research repo.";
    return;
  }
  button.disabled = true;
  log.textContent = "Starting local benchmark... this loads the detector/SAM path and may take a minute.";
  try {
    const body = state.uploadedPath ? { video: state.uploadedPath } : {};
    const response = await fetch("/api/run-benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `benchmark failed: ${response.status}`);
    const best = payload.results[0];
    log.textContent = [
      `Live benchmark complete: ${best.case}`,
      `FPS: ${fmt(best.fps)} | p95: ${fmt(best.frame_ms_p95)} ms | active: ${fmt(best.active_frame_pct, 1)}%`,
      `Avg tracks: ${fmt(best.avg_tracks)} | SAM calls: ${best.sam_calls} | target pass: ${best.meets_target}`,
      "",
      payload.command.join(" "),
    ].join("\n");
    setText("fps", fmt(best.fps));
    setText("p95", fmt(best.frame_ms_p95));
  } catch (error) {
    log.textContent = String(error.message || error);
  } finally {
    button.disabled = false;
  }
}

async function processUploadOnServer() {
  const log = byId("benchmark-log");
  const button = byId("process-upload");
  if (!state.uploadedPath) {
    log.textContent = "Upload a clip first.";
    return;
  }
  button.disabled = true;
  log.textContent = "Processing uploaded clip on the deployable CPU preview path...";
  try {
    const response = await fetch("/api/process-upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: state.uploadedPath, max_frames: 180 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `processing failed: ${response.status}`);
    showVideo(payload.output_video);
    const metrics = payload.metrics;
    log.textContent = [
      "Server CPU preview complete.",
      `Processing FPS: ${fmt(metrics.processing_fps)} | active frames: ${fmt(metrics.active_frame_pct, 1)}%`,
      `Max tracks seen: ${metrics.max_tracks_seen} | total IDs: ${metrics.total_track_ids}`,
      "",
      payload.command.join(" "),
    ].join("\n");
  } catch (error) {
    log.textContent = String(error.message || error);
  } finally {
    button.disabled = false;
  }
}

function showVideo(src) {
  const video = byId("demo-video");
  const image = byId("demo-frame");
  const canvas = byId("browser-canvas");
  const isImage = src.endsWith(".jpg") || src.endsWith(".png");
  document.querySelectorAll(".seg-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.video === src);
  });
  if (isImage) {
    image.src = src;
    image.hidden = false;
    video.hidden = true;
    canvas.hidden = true;
  } else {
    if (video.currentSrc !== new URL(src, window.location.href).href) {
      video.src = src;
    }
    video.hidden = false;
    image.hidden = true;
    canvas.hidden = true;
    state.sourceVideoUrl = src;
  }
}

async function uploadFile(file) {
  const video = byId("demo-video");
  const image = byId("demo-frame");
  const command = byId("upload-command");
  const objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  state.sourceVideoUrl = objectUrl;
  video.hidden = false;
  image.hidden = true;
  setText("feed-title", file.name);
  command.textContent = "Uploading clip to local demo server...";

  const form = new FormData();
  form.append("video", file);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) {
    command.textContent = payload.error || "Upload failed.";
    return;
  }
  state.uploadedPath = payload.path;
  command.textContent = payload.command.join(" ");
}

function letterboxToTensor(video, size = 480) {
  const scratch = document.createElement("canvas");
  scratch.width = size;
  scratch.height = size;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / video.videoWidth, size / video.videoHeight);
  const drawW = Math.round(video.videoWidth * scale);
  const drawH = Math.round(video.videoHeight * scale);
  const padX = Math.floor((size - drawW) / 2);
  const padY = Math.floor((size - drawH) / 2);
  ctx.drawImage(video, padX, padY, drawW, drawH);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  for (let i = 0, px = 0; i < data.length; i += 4, px += 1) {
    tensor[px] = data[i] / 255;
    tensor[size * size + px] = data[i + 1] / 255;
    tensor[2 * size * size + px] = data[i + 2] / 255;
  }
  return {
    input: new ort.Tensor("float32", tensor, [1, 3, size, size]),
    scale,
    padX,
    padY,
  };
}

function parseYoloNms(output, transform, video, minConf = 0.28) {
  const data = output.data;
  const dims = output.dims;
  const rows = dims[dims.length - 2] || 300;
  const cols = dims[dims.length - 1] || 6;
  const detections = [];
  for (let i = 0; i < rows; i += 1) {
    const offset = i * cols;
    const score = Number(data[offset + 4]);
    const cls = Number(data[offset + 5]);
    if (score < minConf || Math.round(cls) !== 0) continue;
    const x1 = (Number(data[offset]) - transform.padX) / transform.scale;
    const y1 = (Number(data[offset + 1]) - transform.padY) / transform.scale;
    const x2 = (Number(data[offset + 2]) - transform.padX) / transform.scale;
    const y2 = (Number(data[offset + 3]) - transform.padY) / transform.scale;
    detections.push({
      score,
      x1: Math.max(0, Math.min(video.videoWidth, x1)),
      y1: Math.max(0, Math.min(video.videoHeight, y1)),
      x2: Math.max(0, Math.min(video.videoWidth, x2)),
      y2: Math.max(0, Math.min(video.videoHeight, y2)),
    });
  }
  return detections.slice(0, 12);
}

async function getYoloSession() {
  if (!window.ort) {
    throw new Error("ONNXRuntime Web did not load. Check network access or bundle ort.min.js for offline deploys.");
  }
  if (!state.yoloSession) {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    state.yoloSession = await ort.InferenceSession.create("/media/models/yolov8n_480_nms.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  return state.yoloSession;
}

async function runBrowserOnnx() {
  const button = byId("run-browser-onnx");
  const log = byId("benchmark-log");
  const video = byId("demo-video");
  const image = byId("demo-frame");
  const canvas = byId("browser-canvas");
  const ctx = canvas.getContext("2d");
  button.disabled = true;
  state.browserRunning = true;
  try {
    const session = await getYoloSession();
    image.hidden = true;
    video.hidden = true;
    canvas.hidden = false;
    const probe = document.createElement("video");
    probe.src = state.sourceVideoUrl;
    probe.muted = true;
    probe.playsInline = true;
    probe.crossOrigin = "anonymous";
    await probe.play().catch(() => undefined);
    if (!probe.videoWidth) {
      await new Promise((resolve) => probe.addEventListener("loadedmetadata", resolve, { once: true }));
    }
    canvas.width = probe.videoWidth;
    canvas.height = probe.videoHeight;
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    let frames = 0;
    const started = performance.now();
    const loop = async () => {
      if (!state.browserRunning || probe.ended || frames >= 120) {
        const elapsed = (performance.now() - started) / 1000;
        log.textContent += `\nBrowser ONNX complete: ${frames} sampled frames at ${fmt(frames / elapsed)} FPS.`;
        button.disabled = false;
        return;
      }
      const t0 = performance.now();
      const transform = letterboxToTensor(probe);
      const outputs = await session.run({ [inputName]: transform.input });
      const detections = parseYoloNms(outputs[outputName], transform, probe);
      ctx.drawImage(probe, 0, 0, canvas.width, canvas.height);
      ctx.lineWidth = Math.max(2, canvas.width / 420);
      ctx.font = `${Math.max(16, canvas.width / 44)}px ui-sans-serif`;
      detections.forEach((det, index) => {
        const color = ["#59c36a", "#5fb7d8", "#e2b95d", "#d17171"][index % 4];
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        ctx.fillText(`person ${det.score.toFixed(2)}`, det.x1 + 4, Math.max(18, det.y1 - 6));
      });
      frames += 1;
      const ms = performance.now() - t0;
      log.textContent = `Browser ONNX detector running on CPU/WASM\nFrame: ${frames} | detections: ${detections.length} | last inference: ${fmt(ms)} ms`;
      setTimeout(loop, 0);
    };
    loop();
  } catch (error) {
    state.browserRunning = false;
    button.disabled = false;
    log.textContent = String(error.message || error);
  }
}

function wireUpload() {
  const input = byId("upload-video");
  const zone = byId("drop-zone");
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (file) uploadFile(file);
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
}

document.querySelectorAll(".seg-button").forEach((button) => {
  button.addEventListener("click", () => showVideo(button.dataset.video));
});
byId("run-benchmark").addEventListener("click", runBenchmark);
byId("run-browser-onnx").addEventListener("click", runBrowserOnnx);
byId("process-upload").addEventListener("click", processUploadOnServer);
wireUpload();
loadSummary().catch((error) => {
  byId("benchmark-log").textContent = String(error.message || error);
});
