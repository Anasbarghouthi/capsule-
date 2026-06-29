const USE_MODEL_API = true;
const MODEL_API_ENDPOINT = "/predict";

const state = {
  file: null,
  folderFiles: [],
  batchResults: [],
  previewUrl: null,
  imageLoaded: false,
  busy: false,
};

const imageInput = document.getElementById("imageInput");
const folderInput = document.getElementById("folderInput");
const chooseButton = document.getElementById("chooseButton");
const folderButton = document.getElementById("folderButton");
const analyzeButton = document.getElementById("analyzeButton");
const analyzeFolderButton = document.getElementById("analyzeFolderButton");
const downloadCsvButton = document.getElementById("downloadCsvButton");
const resetButton = document.getElementById("resetButton");
const dropZone = document.getElementById("dropZone");
const emptyState = document.getElementById("emptyState");
const previewImage = document.getElementById("previewImage");
const overlayLayer = document.getElementById("overlayLayer");
const verdict = document.getElementById("verdict");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const analysisStatus = document.getElementById("analysisStatus");
const detectionList = document.getElementById("detectionList");
const batchSummary = document.getElementById("batchSummary");
const batchProgressBar = document.getElementById("batchProgressBar");
const batchResults = document.getElementById("batchResults");
const modelState = document.getElementById("modelState");

modelState.textContent = USE_MODEL_API ? "Model API active" : "Prototype mode";

chooseButton.addEventListener("click", () => imageInput.click());
folderButton.addEventListener("click", () => folderInput.click());

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    loadFile(file);
  }
});

folderInput.addEventListener("change", (event) => {
  const files = Array.from(event.target.files)
    .filter((file) => file.type.startsWith("image/"))
    .sort((first, second) => getDisplayPath(first).localeCompare(getDisplayPath(second)));
  loadFolder(files);
});

analyzeButton.addEventListener("click", async () => {
  if (!state.file) {
    return;
  }

  setBusy(true);
  try {
    const result = await analyzeFile(state.file);
    renderResult(result);
  } catch (error) {
    renderError(error);
  } finally {
    setBusy(false);
  }
});

analyzeFolderButton.addEventListener("click", analyzeFolder);
downloadCsvButton.addEventListener("click", downloadBatchCsv);
resetButton.addEventListener("click", resetInterface);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-over");
  });
});

dropZone.addEventListener("drop", (event) => {
  const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("image/"));
  if (files.length > 1) {
    loadFolder(files);
  } else if (files[0]) {
    loadFile(files[0]);
  }
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
});

window.addEventListener("resize", () => {
  const currentDetections = JSON.parse(overlayLayer.dataset.detections || "[]");
  renderDetections(currentDetections);
});

function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    return;
  }

  state.folderFiles = [];
  state.batchResults = [];
  folderInput.value = "";
  batchSummary.textContent = "No folder selected";
  batchProgressBar.style.width = "0%";
  batchResults.innerHTML = "";
  showFile(file, []);

  analysisStatus.textContent = "Ready";
  setVerdict("neutral", "Ready for analysis", "--", 0);
  detectionList.innerHTML = '<span class="muted">No findings yet</span>';
  updateButtons();
}

function loadFolder(files) {
  if (!files.length) {
    return;
  }

  state.folderFiles = files;
  state.batchResults = [];
  imageInput.value = "";
  batchResults.innerHTML = "";
  batchSummary.textContent = `${files.length} images selected`;
  batchProgressBar.style.width = "0%";
  showFile(files[0], []);

  analysisStatus.textContent = "Folder ready";
  setVerdict("neutral", "Ready for folder review", "--", 0);
  detectionList.innerHTML = '<span class="muted">Run folder analysis to review all images</span>';
  updateButtons();
}

function showFile(file, detections) {
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  state.imageLoaded = false;

  previewImage.onload = () => {
    state.imageLoaded = true;
    renderDetections(detections);
    updateButtons();
  };

  previewImage.src = state.previewUrl;
  previewImage.hidden = false;
  emptyState.hidden = true;
  overlayLayer.innerHTML = "";
  overlayLayer.dataset.detections = JSON.stringify(detections);

  fileName.textContent = getDisplayPath(file);
  fileSize.textContent = formatBytes(file.size);
  resetButton.disabled = false;
}

async function analyzeFolder() {
  if (!state.folderFiles.length) {
    return;
  }

  setBusy(true);
  state.batchResults = [];
  batchResults.innerHTML = "";
  batchProgressBar.style.width = "0%";
  setVerdict("neutral", "Scanning folder", "--", 0);

  let detectedImages = 0;
  for (let index = 0; index < state.folderFiles.length; index += 1) {
    const file = state.folderFiles[index];
    const progress = Math.round((index / state.folderFiles.length) * 100);
    analysisStatus.textContent = `Scanning ${index + 1}/${state.folderFiles.length}`;
    batchSummary.textContent = `Scanning ${index + 1} of ${state.folderFiles.length}`;
    batchProgressBar.style.width = `${progress}%`;

    try {
      const result = await analyzeFile(file);
      const item = { file, result, error: null };
      state.batchResults.push(item);
      if ((result.detections || []).length) {
        detectedImages += 1;
      }
      appendBatchResult(item, state.batchResults.length - 1);
    } catch (error) {
      const item = { file, result: null, error };
      state.batchResults.push(item);
      appendBatchResult(item, state.batchResults.length - 1);
    }
  }

  batchProgressBar.style.width = "100%";
  batchSummary.textContent = `${detectedImages}/${state.folderFiles.length} images contain findings`;
  analysisStatus.textContent = "Folder review complete";

  const firstDetectedIndex = state.batchResults.findIndex(
    (item) => item.result && (item.result.detections || []).length > 0,
  );
  const indexToShow = firstDetectedIndex >= 0 ? firstDetectedIndex : 0;
  showBatchResult(indexToShow);

  setBusy(false);
}

async function analyzeFile(file) {
  return USE_MODEL_API ? analyzeWithModelApi(file) : analyzeWithPrototype(file);
}

async function analyzeWithPrototype(file) {
  await sleep(650);

  const score = prototypeScore(file);
  const hasFinding = score >= 0.58;
  const confidence = Math.round(score * 100);

  return {
    mode: "prototype",
    verdict: hasFinding ? "Polyp / tumor suspected" : "No clear suspected polyp",
    level: hasFinding ? "danger" : "clear",
    confidence,
    detections: hasFinding
      ? [
          {
            label: "polyp",
            confidence,
            box: {
              x: 0.34,
              y: 0.28,
              width: 0.32,
              height: 0.34,
            },
          },
        ]
      : [],
  };
}

async function analyzeWithModelApi(file) {
  const response = await fetch(MODEL_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
    },
    body: file,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Model API failed with status ${response.status}`);
  }

  return payload;
}

function renderResult(result) {
  const confidence = Number(result.confidence || 0);
  setVerdict(result.level || "neutral", result.verdict || "No result", `${confidence}%`, confidence);
  analysisStatus.textContent = result.mode === "prototype" ? "Prototype result" : "Model result";
  renderDetections(result.detections || []);
}

function renderDetections(detections) {
  overlayLayer.innerHTML = "";
  overlayLayer.dataset.detections = JSON.stringify(detections);

  if (!detections.length) {
    detectionList.innerHTML = '<span class="muted">No findings detected</span>';
    return;
  }

  detectionList.innerHTML = "";
  detections.forEach((detection, index) => {
    const item = document.createElement("div");
    item.className = "detection-item stacked";
    item.innerHTML = `
      <div>
        <strong>${escapeHtml(detection.label || "polyp")} #${index + 1}</strong>
        <small>${escapeHtml(formatBox(detection.box))}</small>
      </div>
      <span>${Math.round(Number(detection.confidence || 0))}%</span>
    `;
    detectionList.appendChild(item);

    if (detection.box) {
      drawBox(detection);
    }
  });
}

function drawBox(detection) {
  const box = detection.box;
  const zoneRect = dropZone.getBoundingClientRect();
  const imageRect = previewImage.getBoundingClientRect();
  const left = imageRect.left - zoneRect.left + Number(box.x) * imageRect.width;
  const top = imageRect.top - zoneRect.top + Number(box.y) * imageRect.height;
  const width = Number(box.width) * imageRect.width;
  const height = Number(box.height) * imageRect.height;
  const element = document.createElement("div");
  element.className = "mock-box";
  element.style.left = `${left}px`;
  element.style.top = `${top}px`;
  element.style.width = `${width}px`;
  element.style.height = `${height}px`;
  element.innerHTML = `<span>${escapeHtml(detection.label || "polyp")} ${Math.round(Number(detection.confidence || 0))}%</span>`;
  overlayLayer.appendChild(element);
}

function appendBatchResult(item, index) {
  const result = item.result;
  const detections = result ? result.detections || [] : [];
  const row = document.createElement("button");
  row.type = "button";
  row.className = `batch-row ${detections.length ? "has-detection" : ""}`;
  row.dataset.index = String(index);
  row.innerHTML = `
    <span class="batch-name">${escapeHtml(getDisplayPath(item.file))}</span>
    <span class="batch-status">${escapeHtml(batchStatusText(item))}</span>
    <span class="batch-boxes">${escapeHtml(formatDetectionsSummary(detections))}</span>
  `;
  row.addEventListener("click", () => showBatchResult(index));
  batchResults.appendChild(row);
}

function showBatchResult(index) {
  const item = state.batchResults[index];
  if (!item) {
    return;
  }

  document.querySelectorAll(".batch-row").forEach((row) => {
    row.classList.toggle("active", Number(row.dataset.index) === index);
  });

  if (item.error) {
    showFile(item.file, []);
    renderError(item.error);
    return;
  }

  showFile(item.file, item.result.detections || []);
  renderResult(item.result);
}

function downloadBatchCsv() {
  if (!state.batchResults.length) {
    return;
  }

  const rows = [["file", "status", "detection_index", "label", "confidence", "x", "y", "width", "height"]];
  state.batchResults.forEach((item) => {
    if (item.error) {
      rows.push([getDisplayPath(item.file), "error", "", "", "", "", "", "", ""]);
      return;
    }

    const detections = item.result.detections || [];
    if (!detections.length) {
      rows.push([getDisplayPath(item.file), "no_polyp", "", "", "", "", "", "", ""]);
      return;
    }

    detections.forEach((detection, index) => {
      const box = detection.box || {};
      rows.push([
        getDisplayPath(item.file),
        "polyp_detected",
        String(index + 1),
        detection.label || "polyp",
        String(detection.confidence || ""),
        String(box.x ?? ""),
        String(box.y ?? ""),
        String(box.width ?? ""),
        String(box.height ?? ""),
      ]);
    });
  });

  const csv = rows.map((row) => row.map(csvCell).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "folder_polyp_detections.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function batchStatusText(item) {
  if (item.error) {
    return "Error";
  }
  const detections = item.result.detections || [];
  if (!detections.length) {
    return "No polyp";
  }
  return `${detections.length} finding${detections.length > 1 ? "s" : ""}`;
}

function formatDetectionsSummary(detections) {
  if (!detections.length) {
    return "No boxes";
  }
  return detections.map((detection, index) => `#${index + 1}: ${formatBox(detection.box)}`).join(" | ");
}

function formatBox(box) {
  if (!box) {
    return "box unavailable";
  }
  return `x=${toPercent(box.x)}, y=${toPercent(box.y)}, w=${toPercent(box.width)}, h=${toPercent(box.height)}`;
}

function toPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function renderError(error) {
  setVerdict("warning", "Analysis failed", "--", 0);
  analysisStatus.textContent = "Error";
  detectionList.innerHTML = `<span class="muted">${escapeHtml(error.message)}</span>`;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  document.body.classList.toggle("is-busy", isBusy);
  updateButtons();
  analysisStatus.textContent = isBusy ? "Analyzing" : analysisStatus.textContent;
}

function updateButtons() {
  analyzeButton.disabled = state.busy || !state.imageLoaded || !state.file;
  analyzeFolderButton.disabled = state.busy || !state.folderFiles.length;
  downloadCsvButton.disabled = state.busy || !state.batchResults.length;
  chooseButton.disabled = state.busy;
  folderButton.disabled = state.busy;
  resetButton.disabled = state.busy || (!state.file && !state.folderFiles.length);
}

function setVerdict(level, text, confidenceText, confidencePercent) {
  verdict.className = `verdict ${level}`;
  verdict.textContent = text;
  confidenceValue.textContent = confidenceText;
  confidenceBar.style.width = `${Math.max(0, Math.min(confidencePercent, 100))}%`;
}

function resetInterface() {
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.file = null;
  state.folderFiles = [];
  state.batchResults = [];
  state.previewUrl = null;
  state.imageLoaded = false;
  state.busy = false;
  document.body.classList.remove("is-busy");
  imageInput.value = "";
  folderInput.value = "";
  previewImage.removeAttribute("src");
  previewImage.hidden = true;
  emptyState.hidden = false;
  overlayLayer.innerHTML = "";
  overlayLayer.dataset.detections = "[]";
  fileName.textContent = "--";
  fileSize.textContent = "--";
  analysisStatus.textContent = "Ready";
  batchSummary.textContent = "No folder selected";
  batchProgressBar.style.width = "0%";
  batchResults.innerHTML = "";
  setVerdict("neutral", "Awaiting image", "--", 0);
  detectionList.innerHTML = '<span class="muted">No image selected</span>';
  updateButtons();
}

function getDisplayPath(file) {
  return file.webkitRelativePath || file.name;
}

function prototypeScore(file) {
  const text = `${file.name}-${file.size}-${file.lastModified}`;
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return 0.35 + (hash % 5000) / 10000;
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let size = bytes / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function csvCell(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}
