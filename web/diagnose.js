// web/diagnose.js - Leaf disease diagnosis UI logic
// Strictly uses textContent for dynamic server data (no innerHTML)

let selectedFile = null;
let currentResult = null;

// DOM Elements
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const previewImg = document.getElementById("preview-img");
const btnAnalyze = document.getElementById("btn-analyze");
const btnChangePhoto = document.getElementById("btn-change-photo");
const loadingBox = document.getElementById("loading-box");
const resultBox = document.getElementById("result-box");
const btnScanAnother = document.getElementById("btn-scan-another");

// Client-side image resize using HTML5 Canvas (max 1024px)
function resizeImageBeforeUpload(file, maxDimension = 1024) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        let width = img.width;
        let height = img.height;

        if (width > maxDimension || height > maxDimension) {
          if (width > height) {
            height = Math.round((height * maxDimension) / width);
            width = maxDimension;
          } else {
            width = Math.round((width * maxDimension) / height);
            height = maxDimension;
          }
        }

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(new File([blob], file.name, { type: "image/jpeg" }));
            } else {
              resolve(file); // fallback to original
            }
          },
          "image/jpeg",
          0.88
        );
      };
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function handleFileSelection(file) {
  if (!file) return;
  selectedFile = file;

  // Show local preview
  const url = URL.createObjectURL(file);
  previewImg.src = url;

  uploadZone.style.display = "none";
  previewContainer.style.display = "block";
  resultBox.style.display = "none";
}

function resetUploadState() {
  selectedFile = null;
  currentResult = null;
  fileInput.value = "";
  previewContainer.style.display = "none";
  resultBox.style.display = "none";
  loadingBox.style.display = "none";
  uploadZone.style.display = "block";
}

async function performDiagnosis() {
  if (!selectedFile) return;

  // Transition to loading state
  previewContainer.style.display = "none";
  loadingBox.style.display = "block";
  resultBox.style.display = "none";

  try {
    // 1. Client-side resize to max 1024px to save mobile data
    const optimizedFile = await resizeImageBeforeUpload(selectedFile, 1024);

    // 2. Prepare multipart form data
    const formData = new FormData();
    formData.append("file", optimizedFile);

    // 3. API request
    const response = await fetch("/api/diagnose", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP_${response.status}`);
    }

    const data = await response.json();
    currentResult = data;
    renderResult(data);
  } catch (error) {
    console.error("Diagnosis error:", error);
    renderErrorResult(error);
  } finally {
    loadingBox.style.display = "none";
  }
}

function renderResult(data) {
  const lang = window.getCurrentLang();
  resultBox.className = "result-box"; // reset classes
  resultBox.style.display = "block";

  const statusBadge = document.getElementById("status-badge");
  const resultTitle = document.getElementById("result-main-title");
  const resultDesc = document.getElementById("result-main-desc");
  const reliabilityBanner = document.getElementById("reliability-banner");
  const symptomsBlock = document.getElementById("symptoms-block");
  const symptomsText = document.getElementById("symptoms-text");
  const adviceBlock = document.getElementById("advice-block");
  const adviceText = document.getElementById("advice-text");
  const topPredictionsList = document.getElementById("predictions-list");

  // Empty predictions list safely
  topPredictionsList.textContent = "";

  // 1. Handle Status Appearance
  if (data.status === "ok") {
    resultBox.classList.add("status-ok");
    statusBadge.textContent = window.t("status_ok_title");
    resultDesc.textContent = window.t("status_ok_desc");
  } else if (data.status === "uncertain") {
    resultBox.classList.add("status-uncertain");
    statusBadge.textContent = window.t("status_uncertain_title");
    resultDesc.textContent = window.t("status_uncertain_desc");
  } else if (data.status === "unknown") {
    resultBox.classList.add("status-unknown");
    statusBadge.textContent = window.t("status_unknown_title");
    resultDesc.textContent = window.t("status_unknown_desc");
  } else {
    resultBox.classList.add("status-error");
    statusBadge.textContent = window.t("status_bad_image_title");
    const errKey = data.error_code ? `err_${data.error_code}` : "err_bad_image";
    resultDesc.textContent = window.t(errKey, window.t("status_bad_image_desc"));
  }

  // 2. Main Title (Crop & Disease Name)
  if (data.knowledge) {
    const crop = data.knowledge.crop_name ? data.knowledge.crop_name[lang] || data.knowledge.crop_name["en"] : "";
    const disease = data.knowledge.disease_name ? data.knowledge.disease_name[lang] || data.knowledge.disease_name["en"] : "";
    resultTitle.textContent = `${crop} — ${disease}`;
  } else if (data.top_label) {
    resultTitle.textContent = data.top_label.replace("___", " - ");
  } else {
    resultTitle.textContent = window.t("status_bad_image_title");
  }

  // 3. Reliability Warning
  if (data.reliability && !data.reliability.is_reliable) {
    reliabilityBanner.style.display = "block";
    const warnTitle = document.getElementById("reliability-title");
    const warnDesc = document.getElementById("reliability-desc");
    warnTitle.textContent = window.t("reliability_warning_title");
    if (data.reliability.reason === "lab_only") {
      warnDesc.textContent = window.t("reliability_lab_only_desc");
    } else {
      warnDesc.textContent = window.t("reliability_low_recall_desc");
    }
  } else {
    reliabilityBanner.style.display = "none";
  }

  // 4. Symptoms & Advice
  if (data.knowledge && data.status !== "bad_image") {
    symptomsBlock.style.display = "block";
    adviceBlock.style.display = "block";
    symptomsText.textContent = data.knowledge.symptoms ? data.knowledge.symptoms[lang] || data.knowledge.symptoms["en"] : "";
    adviceText.textContent = data.knowledge.cultural_advice ? data.knowledge.cultural_advice[lang] || data.knowledge.cultural_advice["en"] : "";
  } else {
    symptomsBlock.style.display = "none";
    adviceBlock.style.display = "none";
  }

  // 5. Render Top-3 Predictions Bars safely
  if (data.predictions && data.predictions.length > 0) {
    document.getElementById("top-predictions-title").style.display = "block";
    data.predictions.forEach((pred) => {
      const row = document.createElement("div");
      row.className = "prediction-row";

      const meta = document.createElement("div");
      meta.className = "pred-meta";

      const nameSpan = document.createElement("span");
      nameSpan.textContent = pred.label.replace("___", " - ");

      const pct = Math.round(pred.probability * 1000) / 10;
      const probSpan = document.createElement("span");
      probSpan.textContent = `${pct}%`;

      meta.appendChild(nameSpan);
      meta.appendChild(probSpan);

      const track = document.createElement("div");
      track.className = "progress-track";

      const fill = document.createElement("div");
      fill.className = "progress-fill";
      fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;

      track.appendChild(fill);
      row.appendChild(meta);
      row.appendChild(track);

      topPredictionsList.appendChild(row);
    });
  } else {
    document.getElementById("top-predictions-title").style.display = "none";
  }
}

function renderErrorResult(err) {
  resultBox.className = "result-box status-error";
  resultBox.style.display = "block";

  document.getElementById("status-badge").textContent = window.t("status_bad_image_title");
  document.getElementById("result-main-title").textContent = window.t("err_network_error");
  document.getElementById("result-main-desc").textContent = err.message || window.t("err_internal_server_error");
  document.getElementById("reliability-banner").style.display = "none";
  document.getElementById("symptoms-block").style.display = "none";
  document.getElementById("advice-block").style.display = "none";
  document.getElementById("top-predictions-title").style.display = "none";
  document.getElementById("predictions-list").textContent = "";
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  if (!uploadZone) return;

  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  btnAnalyze.addEventListener("click", performDiagnosis);
  btnChangePhoto.addEventListener("click", resetUploadState);
  btnScanAnother.addEventListener("click", resetUploadState);

  // Re-render when language changes
  window.addEventListener("languageChanged", () => {
    if (currentResult) {
      renderResult(currentResult);
    }
  });
});
