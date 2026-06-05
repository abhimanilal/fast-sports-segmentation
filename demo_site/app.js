const demoAssets = {
  recLeague: {
    raw: "/media/rec_league_raw_15s.mp4",
    mask: "/media/rec_league_masks_15s.mp4",
    poster: "/media/rec_league_poster.jpg",
    rawTitle: "Raw rec-league clip",
    maskTitle: "On-court YOLO-Seg tracking",
    note: "Rec-league sample loaded. Mask overlay is pre-rendered; Analyze current frame runs YOLO ONNX in this browser.",
  },
  pickup5v5: {
    raw: "/media/pickup_5v5_raw_12s.mp4",
    mask: "/media/pickup_5v5_masks_12s.mp4",
    poster: "/media/pickup_5v5_poster.jpg",
    rawTitle: "Raw pickup 5v5 clip",
    maskTitle: "Pickup dense YOLO-Seg tracking",
    note: "Pickup 5v5 sample loaded. Mask overlay is pre-rendered; Analyze current frame runs YOLO ONNX in this browser.",
  },
  streetballPov: {
    raw: "/media/streetball_pov_raw_12s.mp4",
    mask: "/media/streetball_pov_masks_12s.mp4",
    poster: "/media/streetball_pov_poster.jpg",
    rawTitle: "Raw streetball POV clip",
    maskTitle: "Streetball dense YOLO-Seg tracking",
    note: "Streetball POV sample loaded. Mask overlay is pre-rendered; Analyze current frame runs YOLO ONNX in this browser.",
  },
};

const state = {
  summary: null,
  source: "sample",
  view: "raw",
  demo: "recLeague",
  uploadedPath: null,
  sourceVideoUrl: demoAssets.recLeague.raw,
  seedPrompt: "basketball players on court",
  yoloSession: null,
};

const byId = (id) => document.getElementById(id);

function fmt(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function setText(id, value) {
  const node = byId(id);
  if (node) node.textContent = value;
}

function setRunState(value) {
  setText("run-state", value);
}

function setLog(message) {
  setText("benchmark-log", message);
}

function currentDemo() {
  return demoAssets[state.demo] || demoAssets.recLeague;
}

function setVideo(src, title = "Video input", poster = "") {
  const video = byId("demo-video");
  const iframe = byId("youtube-frame");
  const canvas = byId("browser-canvas");
  const currentTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
  iframe.hidden = true;
  canvas.hidden = true;
  video.hidden = false;
  video.poster = poster;
  if (video.currentSrc !== new URL(src, window.location.href).href) {
    video.src = src;
    video.addEventListener(
      "loadedmetadata",
      () => {
        video.currentTime = Math.min(currentTime, Math.max(0, video.duration - 0.2));
      },
      { once: true },
    );
  }
  state.sourceVideoUrl = src;
  setText("feed-title", title);
}

function setDemo(demoKey) {
  if (!demoAssets[demoKey]) return;
  state.demo = demoKey;
  document.querySelectorAll(".demo-option").forEach((button) => {
    button.classList.toggle("active", button.dataset.demo === demoKey);
  });
  if (state.source !== "sample") {
    setSource("sample");
    return;
  }
  setView(state.view);
  setLog(currentDemo().note);
}

function setSource(source) {
  state.source = source;
  document.querySelectorAll(".source-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === source);
  });
  byId("youtube-controls").hidden = source !== "youtube";
  byId("drop-zone").style.display = source === "youtube" ? "none" : "flex";
  byId("process-upload").disabled = source !== "upload" || !state.uploadedPath;
  document.querySelectorAll(".view-button").forEach((button) => {
    button.disabled = source !== "sample";
  });

  if (source === "sample") {
    setRunState("Ready");
    setView(state.view);
    setLog(currentDemo().note);
  } else if (source === "youtube") {
    setRunState("Embed mode");
    setLog("Paste a YouTube URL to embed it for review. Browser inference is disabled for YouTube frames because they are cross-origin.");
  } else {
    setRunState("Upload mode");
    if (!state.uploadedPath) {
      setLog("Choose an MP4/WebM file to preview, then analyze a frame or render a server preview.");
    }
  }
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".view-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  if (state.source === "sample") {
    const demo = currentDemo();
    setVideo(view === "mask" ? demo.mask : demo.raw, view === "mask" ? demo.maskTitle : demo.rawTitle, demo.poster);
  }
}

function youtubeEmbedUrl(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl.trim());
  } catch {
    return null;
  }
  let id = "";
  if (parsed.hostname.includes("youtu.be")) {
    id = parsed.pathname.replace("/", "");
  } else if (parsed.hostname.includes("youtube.com")) {
    if (parsed.pathname.startsWith("/embed/")) {
      id = parsed.pathname.split("/embed/")[1].split("/")[0];
    } else if (parsed.pathname.startsWith("/shorts/")) {
      id = parsed.pathname.split("/shorts/")[1].split("/")[0];
    } else {
      id = parsed.searchParams.get("v") || "";
    }
  }
  return id ? `https://www.youtube.com/embed/${encodeURIComponent(id)}` : null;
}

function loadYouTube() {
  const url = byId("youtube-url").value;
  const embed = youtubeEmbedUrl(url);
  if (!embed) {
    setLog("That does not look like a YouTube watch, shorts, youtu.be, or embed URL.");
    return;
  }
  const video = byId("demo-video");
  const canvas = byId("browser-canvas");
  const iframe = byId("youtube-frame");
  video.hidden = true;
  canvas.hidden = true;
  iframe.hidden = false;
  iframe.src = embed;
  setText("feed-title", "Embedded YouTube clip");
  setRunState("YouTube loaded");
  setLog("YouTube clip embedded. Use it for visual review or upload a local clip to run model inference.");
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
  if (!response.ok) return;
  renderSummary(await response.json());
}

async function processUploadOnServer() {
  if (!state.uploadedPath) {
    setLog("Choose a local video before rendering a server preview.");
    return;
  }
  const button = byId("process-upload");
  button.disabled = true;
  setRunState("Rendering");
  setLog(`Rendering a CPU preview using seed prompt: "${state.seedPrompt}"`);
  try {
    const response = await fetch("/api/process-upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: state.uploadedPath, max_frames: 180, prompt: state.seedPrompt }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error("processing failed");
    setVideo(payload.output_video, "Server preview overlay");
    const metrics = payload.metrics;
    setRunState("Preview ready");
    setLog(`Server preview complete. ${fmt(metrics.processing_fps)} FPS CPU preview, ${fmt(metrics.active_frame_pct, 1)}% active frames.`);
  } catch {
    setRunState("Upload mode");
    setLog("Server preview could not finish for this clip. Try a shorter MP4/WebM or analyze a frame in the browser.");
  } finally {
    button.disabled = false;
  }
}

async function uploadFile(file) {
  const objectUrl = URL.createObjectURL(file);
  setSource("upload");
  setVideo(objectUrl, file.name);
  setRunState("Video selected");
  byId("upload-caption").textContent = `${file.name} is ready.`;
  setLog("Local preview ready. Analyze the current frame or render a server preview.");
  setText("upload-command", "Uploading clip to prepare a reproducible local command.");

  const form = new FormData();
  form.append("video", file);
  try {
    const response = await fetch("/api/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error("upload failed");
    state.uploadedPath = payload.path;
    byId("process-upload").disabled = false;
    setText("upload-command", `${payload.command.join(" ")} --categories "${state.seedPrompt}"`);
  } catch {
    setText("upload-command", "Server upload is unavailable. Browser frame analysis can still run on the selected local video.");
  }
}

function letterboxToTensor(source, size = 480) {
  const scratch = document.createElement("canvas");
  scratch.width = size;
  scratch.height = size;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  const srcW = source.videoWidth || source.width;
  const srcH = source.videoHeight || source.height;
  const scale = Math.min(size / srcW, size / srcH);
  const drawW = Math.round(srcW * scale);
  const drawH = Math.round(srcH * scale);
  const padX = Math.floor((size - drawW) / 2);
  const padY = Math.floor((size - drawH) / 2);
  ctx.drawImage(source, padX, padY, drawW, drawH);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  for (let i = 0, px = 0; i < data.length; i += 4, px += 1) {
    tensor[px] = data[i] / 255;
    tensor[size * size + px] = data[i + 1] / 255;
    tensor[2 * size * size + px] = data[i + 2] / 255;
  }
  return { input: new ort.Tensor("float32", tensor, [1, 3, size, size]), scale, padX, padY };
}

function parseYoloNms(output, transform, width, height, minConf = 0.28) {
  const data = output.data;
  const rows = output.dims[output.dims.length - 2] || 300;
  const cols = output.dims[output.dims.length - 1] || 6;
  const detections = [];
  for (let i = 0; i < rows; i += 1) {
    const offset = i * cols;
    const score = Number(data[offset + 4]);
    const cls = Number(data[offset + 5]);
    if (score < minConf || Math.round(cls) !== 0) continue;
    detections.push({
      score,
      x1: Math.max(0, Math.min(width, (Number(data[offset]) - transform.padX) / transform.scale)),
      y1: Math.max(0, Math.min(height, (Number(data[offset + 1]) - transform.padY) / transform.scale)),
      x2: Math.max(0, Math.min(width, (Number(data[offset + 2]) - transform.padX) / transform.scale)),
      y2: Math.max(0, Math.min(height, (Number(data[offset + 3]) - transform.padY) / transform.scale)),
    });
  }
  return detections.slice(0, 12);
}

async function getYoloSession() {
  if (!window.ort) throw new Error("onnxruntime unavailable");
  if (!state.yoloSession) {
    setRunState("Loading model");
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    state.yoloSession = await ort.InferenceSession.create("/media/models/yolov8n_480_nms.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  return state.yoloSession;
}

async function analyzeCurrentFrame() {
  if (state.source === "youtube") {
    setLog("YouTube embeds are cross-origin, so this browser cannot read their frames. Upload a clip or use the sample video for inference.");
    return;
  }
  const video = byId("demo-video");
  if (!video.videoWidth) {
    setLog("Wait for the video to load before analyzing a frame.");
    return;
  }
  const button = byId("run-browser-onnx");
  button.disabled = true;
  setRunState("Analyzing");
  setLog(`Analyzing one frame with seed prompt: "${state.seedPrompt}"`);
  try {
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const session = await getYoloSession();
    const transform = letterboxToTensor(video);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const started = performance.now();
    const outputs = await session.run({ [inputName]: transform.input });
    const elapsed = performance.now() - started;
    const detections = parseYoloNms(outputs[outputName], transform, video.videoWidth, video.videoHeight);
    const canvas = byId("browser-canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = Math.max(2, canvas.width / 360);
    ctx.font = `${Math.max(15, canvas.width / 42)}px ui-sans-serif`;
    detections.forEach((det, index) => {
      const color = ["#69d16f", "#6fbfe0", "#d8b861", "#dc766e"][index % 4];
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
      ctx.fillText(`person ${det.score.toFixed(2)}`, det.x1 + 4, Math.max(18, det.y1 - 6));
    });
    video.hidden = true;
    byId("youtube-frame").hidden = true;
    canvas.hidden = false;
    setRunState("Frame analyzed");
    setLog(`Detected ${detections.length} people in ${fmt(elapsed)} ms. Full video rendering is handled by the server preview so the browser stays responsive.`);
  } catch {
    setRunState("Ready");
    setLog("Browser frame analysis could not start. Try the sample clip, a shorter upload, or server preview.");
  } finally {
    button.disabled = false;
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

document.querySelectorAll(".source-tab").forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});
document.querySelectorAll(".view-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
document.querySelectorAll(".demo-option").forEach((button) => {
  button.addEventListener("click", () => setDemo(button.dataset.demo));
});
byId("apply-prompt").addEventListener("click", () => {
  state.seedPrompt = byId("seed-prompt").value.trim() || "person";
  setLog(`Seed prompt set to "${state.seedPrompt}".`);
});
byId("load-youtube").addEventListener("click", loadYouTube);
byId("run-browser-onnx").addEventListener("click", analyzeCurrentFrame);
byId("process-upload").addEventListener("click", processUploadOnServer);
wireUpload();
setSource("sample");
loadSummary();
