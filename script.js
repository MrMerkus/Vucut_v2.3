// ══════════════════════════════════════════════════════════════════════════════
// script.js  —  Vücut + El + Yüz Takip Sistemi (Birleşik)
// MediaPipe Tasks Vision v0.10.3
// ══════════════════════════════════════════════════════════════════════════════

import {
  PoseLandmarker,
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.js";

// ── DOM ───────────────────────────────────────────────────────────────────────
const video         = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx     = canvasElement.getContext("2d");
const drawingUtils  = new DrawingUtils(canvasCtx);

const webcamButton      = document.getElementById("webcamButton");
const btnText           = document.getElementById("btnText");
const screenshotBtn     = document.getElementById("screenshotButton");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const modelStatus       = document.getElementById("model-status");
const fpsDisplay        = document.getElementById("fps-display");
const personCountEl     = document.getElementById("person-count");
const metricsPanel      = document.getElementById("metrics-panel");
const noPersonMsg       = document.getElementById("no-person-msg");

// ── KİŞİ RENKLERİ ─────────────────────────────────────────────────────────────
const PERSON_COLORS = [
  { hex: "#00e5cc", name: "KİŞİ 1" },
  { hex: "#ff4081", name: "KİŞİ 2" },
  { hex: "#ffd600", name: "KİŞİ 3" },
  { hex: "#69ff47", name: "KİŞİ 4" },
  { hex: "#e040fb", name: "KİŞİ 5" },
  { hex: "#ff6d00", name: "KİŞİ 6" },
];

// ── STATE ──────────────────────────────────────────────────────────────────────
let poseLandmarker, handLandmarker, faceLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;
let fpsFrameCount = 0;
let fpsLastTime   = performance.now();
let modelsReady   = false;

// ── KİŞİ STABİLİZASYON SİSTEMİ ───────────────────────────────────────────────
const MAX_PERSONS  = 6;
const MAX_DIST     = 0.25;
const LOST_TIMEOUT = 45;

const trackerSlots = Array.from({ length: MAX_PERSONS }, () => ({
  active: false, cx: 0, cy: 0, age: 0,
}));

function getPoseCentroid(lm) {
  const pts = [lm[11], lm[12], lm[23], lm[24]];
  return {
    cx: pts.reduce((s, p) => s + p.x, 0) / pts.length,
    cy: pts.reduce((s, p) => s + p.y, 0) / pts.length,
  };
}

function matchPosesToSlots(landmarks) {
  const n          = landmarks.length;
  const centroids  = landmarks.map(getPoseCentroid);
  const poseToSlot = new Array(n).fill(-1);
  const slotUsed   = new Array(MAX_PERSONS).fill(false);

  const costs = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < MAX_PERSONS; j++) {
      if (!trackerSlots[j].active) continue;
      const d = Math.hypot(
        centroids[i].cx - trackerSlots[j].cx,
        centroids[i].cy - trackerSlots[j].cy
      );
      costs.push({ i, j, d });
    }
  }
  costs.sort((a, b) => a.d - b.d);

  for (const { i, j, d } of costs) {
    if (poseToSlot[i] !== -1 || slotUsed[j]) continue;
    if (d < MAX_DIST) { poseToSlot[i] = j; slotUsed[j] = true; }
  }

  for (let i = 0; i < n; i++) {
    if (poseToSlot[i] !== -1) continue;
    for (let j = 0; j < MAX_PERSONS; j++) {
      if (!trackerSlots[j].active && !slotUsed[j]) {
        poseToSlot[i] = j; slotUsed[j] = true; break;
      }
    }
  }

  const seenSlots = new Set(poseToSlot.filter(j => j !== -1));
  for (let j = 0; j < MAX_PERSONS; j++) {
    if (seenSlots.has(j)) {
      const pi = poseToSlot.indexOf(j);
      trackerSlots[j].active = true;
      trackerSlots[j].cx     = centroids[pi].cx;
      trackerSlots[j].cy     = centroids[pi].cy;
      trackerSlots[j].age    = 0;
    } else if (trackerSlots[j].active) {
      trackerSlots[j].age++;
      if (trackerSlots[j].age > LOST_TIMEOUT) {
        trackerSlots[j].active = false;
        trackerSlots[j].age    = 0;
      }
    }
  }
  return poseToSlot;
}

// ── MODEL KURULUMU ─────────────────────────────────────────────────────────────
// Modeller yüklenene kadar butonu pasif yap
webcamButton.disabled = true;
webcamButton.style.opacity = "0.5";
webcamButton.style.cursor  = "not-allowed";

function setLoadingStep(text, isError = false) {
  modelStatus.textContent = isError ? `● ${text}` : `⟳ ${text}`;
  if (isError) modelStatus.style.color = "var(--red)";
  else         modelStatus.style.color = "";
  modelStatus.classList.remove("ready");
}

async function setupModels() {
  try {
    setLoadingStep("WASM YÜKLENİYOR...");

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );

    // GPU'yu dene, hata alırsa CPU'ya düş
    async function tryCreate(createFn) {
      try {
        return await createFn("GPU");
      } catch (gpuErr) {
        console.warn("GPU başarısız, CPU'ya geçiliyor:", gpuErr);
        return await createFn("CPU");
      }
    }

    setLoadingStep("VÜCUT MODELİ (1/3)...");
    poseLandmarker = await tryCreate(async (delegate) =>
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate,
        },
        runningMode: "VIDEO",
        numPoses: MAX_PERSONS,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })
    );

    setLoadingStep("EL MODELİ (2/3)...");
    handLandmarker = await tryCreate(async (delegate) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate,
        },
        runningMode: "VIDEO",
        numHands: 4,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })
    );

    setLoadingStep("YÜZ MODELİ (3/3)...");
    faceLandmarker = await tryCreate(async (delegate) =>
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate,
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: MAX_PERSONS,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })
    );

    modelsReady = true;
    modelStatus.textContent = "● MODEL HAZIR";
    modelStatus.classList.add("ready");
    modelStatus.style.color = "";
    webcamButton.disabled = false;
    webcamButton.style.opacity = "";
    webcamButton.style.cursor  = "";

  } catch (err) {
    console.error("Model yükleme hatası:", err);
    setLoadingStep("YÜKLEME BAŞARISIZ — Sayfayı yenileyin", true);
    // Butonu tekrar açıyoruz ki kullanıcı sayfayı yenileyebilsin
    webcamButton.disabled = false;
    webcamButton.style.opacity = "0.5";
  }
}

setupModels();

// ── AÇI HESABI ────────────────────────────────────────────────────────────────
function calcAngle(A, B, C) {
  if (!A || !B || !C) return null;
  const BA = { x: A.x - B.x, y: A.y - B.y };
  const BC = { x: C.x - B.x, y: C.y - B.y };
  const dot  = BA.x * BC.x + BA.y * BC.y;
  const magA = Math.hypot(BA.x, BA.y);
  const magC = Math.hypot(BC.x, BC.y);
  if (magA === 0 || magC === 0) return null;
  const cos = Math.max(-1, Math.min(1, dot / (magA * magC)));
  return Math.round(Math.acos(cos) * (180 / Math.PI));
}

// ── YÜZ METRİKLERİ ───────────────────────────────────────────────────────────
function calcHeadRoll(lm) {
  const l = lm[33], r = lm[263];
  if (!l || !r) return null;
  return Math.round(Math.atan2(r.y - l.y, r.x - l.x) * (180 / Math.PI));
}

function calcHeadYaw(lm) {
  const nose = lm[1], leftEar = lm[234], rightEar = lm[454];
  if (!nose || !leftEar || !rightEar) return null;
  const lD = Math.abs(nose.x - leftEar.x);
  const rD = Math.abs(nose.x - rightEar.x);
  const total = lD + rD;
  return total === 0 ? 0 : Math.round(((lD - rD) / total) * 90);
}

function calcMouthOpen(lm) {
  const top = lm[13], bot = lm[14], left = lm[61], right = lm[291];
  if (!top || !bot || !left || !right) return null;
  const v = Math.hypot(top.x - bot.x, top.y - bot.y);
  const h = Math.hypot(left.x - right.x, left.y - right.y);
  return h > 0 ? Math.round((v / h) * 100) : 0;
}

function getBlendShape(categories, key) {
  const c = categories.find(c => c.categoryName === key);
  return c ? +c.score : 0;
}

function guessExpression(categories) {
  const smile   = (getBlendShape(categories, "mouthSmileLeft") + getBlendShape(categories, "mouthSmileRight")) / 2;
  const blink   = (getBlendShape(categories, "eyeBlinkLeft")   + getBlendShape(categories, "eyeBlinkRight"))  / 2;
  const jawOpen = getBlendShape(categories, "jawOpen");
  const browUp  = getBlendShape(categories, "browInnerUp");
  if (blink   > 0.6) return "GÖZLER KAPALI";
  if (jawOpen > 0.5) return "AĞIZ AÇIK";
  if (smile   > 0.4) return "GÜLÜMSÜYOR";
  if (browUp  > 0.5) return "ŞAŞIRMIŞ";
  return "NÖTR";
}

// ── KİŞİ KARTI HTML ───────────────────────────────────────────────────────────
function createPersonCard(slotIdx) {
  const color = PERSON_COLORS[slotIdx % PERSON_COLORS.length];
  const id    = `p${slotIdx}`;

  const card = document.createElement("div");
  card.className = "person-card";
  card.id        = `person-card-${slotIdx}`;
  card.style.setProperty("--person-color", color.hex);

  card.innerHTML = `
    <div class="person-header">
      <div class="person-dot"></div>
      <div class="person-title">${color.name}</div>
    </div>
    <div class="person-card-body">

      <!-- VÜCUT: Eklem Açıları -->
      <div>
        <div class="panel-title">EKLEM AÇILARI</div>
        <div class="angles-grid">
          <div class="angle-card" id="${id}-card-left-elbow">
            <div class="angle-label">SOL DİRSEK</div>
            <div class="angle-value" id="${id}-angle-left-elbow">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-left-elbow"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-right-elbow">
            <div class="angle-label">SAĞ DİRSEK</div>
            <div class="angle-value" id="${id}-angle-right-elbow">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-right-elbow"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-left-shoulder">
            <div class="angle-label">SOL OMUZ</div>
            <div class="angle-value" id="${id}-angle-left-shoulder">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-left-shoulder"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-right-shoulder">
            <div class="angle-label">SAĞ OMUZ</div>
            <div class="angle-value" id="${id}-angle-right-shoulder">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-right-shoulder"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-left-knee">
            <div class="angle-label">SOL DİZ</div>
            <div class="angle-value" id="${id}-angle-left-knee">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-left-knee"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-right-knee">
            <div class="angle-label">SAĞ DİZ</div>
            <div class="angle-value" id="${id}-angle-right-knee">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-right-knee"></div></div>
          </div>
          <div class="angle-card wide" id="${id}-card-shoulder-spread">
            <div class="angle-label">OMUZ GENİŞLİĞİ AÇISI</div>
            <div class="angle-value" id="${id}-angle-shoulder-spread">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-shoulder-spread"></div></div>
          </div>
          <div class="angle-card wide" id="${id}-card-hip">
            <div class="angle-label">KALÇA AÇISI</div>
            <div class="angle-value" id="${id}-angle-hip">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-hip"></div></div>
          </div>
        </div>
      </div>

      <!-- YÜZ: Analiz (sadece yüz algılandığında görünür) -->
      <div id="${id}-face-section" style="display:none">
        <div class="panel-title">YÜZ ANALİZİ</div>
        <div class="angles-grid">
          <div class="angle-card" id="${id}-card-roll">
            <div class="angle-label">BAŞ EĞİMİ</div>
            <div class="angle-value" id="${id}-val-roll">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-roll"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-yaw">
            <div class="angle-label">BAŞ DÖNÜŞÜ</div>
            <div class="angle-value" id="${id}-val-yaw">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-yaw"></div></div>
          </div>
          <div class="angle-card wide" id="${id}-card-mouth">
            <div class="angle-label">AĞIZ AÇIKLIĞI</div>
            <div class="angle-value" id="${id}-val-mouth">—%</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-mouth"></div></div>
          </div>
        </div>
        <div class="stats-grid" style="margin-top:6px">
          <div class="stat-item">
            <div class="stat-label">YÜZ</div>
            <div class="stat-value yes" id="${id}-stat-face">ALGILANDI</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">İFADE</div>
            <div class="stat-value" id="${id}-stat-expr" style="font-size:9px">—</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">GÜLÜMSEME</div>
            <div class="stat-value" id="${id}-stat-smile">—</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">GÖZ KIRPMA</div>
            <div class="stat-value" id="${id}-stat-blink">—</div>
          </div>
        </div>
      </div>

      <!-- VERİ AKIŞI -->
      <div>
        <div class="panel-title">VERİ AKIŞI</div>
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-label">VÜCUT</div>
            <div class="stat-value yes" id="${id}-stat-pose">ALGILANDI</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">SOL EL</div>
            <div class="stat-value" id="${id}-stat-left-hand">HAYIR</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">SAĞ EL</div>
            <div class="stat-value" id="${id}-stat-right-hand">HAYIR</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">NOKTA SAYISI</div>
            <div class="stat-value" id="${id}-stat-landmarks">0</div>
          </div>
        </div>
      </div>

      <!-- SİMETRİ SKORU -->
      <div>
        <div class="panel-title">SİMETRİ SKORU</div>
        <div class="symmetry-display">
          <div class="symmetry-score" id="${id}-symmetry-score">—</div>
          <div class="symmetry-label">Omuz / Dirsek / Diz simetrisi</div>
          <div class="symmetry-bar-wrap">
            <div class="symmetry-bar" id="${id}-symmetry-bar"></div>
          </div>
        </div>
      </div>

    </div>
  `;
  return card;
}

// ── UI GÜNCELLEYİCİLER ───────────────────────────────────────────────────────
function updateAngleCard(prefix, name, degrees) {
  const valueEl = document.getElementById(`${prefix}-angle-${name}`);
  const barEl   = document.getElementById(`${prefix}-bar-${name}`);
  const card    = document.getElementById(`${prefix}-card-${name}`);
  if (!valueEl) return;
  if (degrees === null) {
    valueEl.textContent = "—°";
    if (barEl) barEl.style.width = "0%";
    if (card)  card.classList.remove("active");
    return;
  }
  valueEl.textContent = `${degrees}°`;
  if (barEl) barEl.style.width = `${Math.min(100, (degrees / 180) * 100)}%`;
  if (card)  card.classList.add("active");
}

function updateStat(prefix, name, isActive) {
  const el = document.getElementById(`${prefix}-stat-${name}`);
  if (!el) return;
  el.textContent = isActive ? "ALGILANDI" : "HAYIR";
  el.className   = "stat-value" + (isActive ? " yes" : "");
}

function updateSymmetry(prefix, score) {
  const el  = document.getElementById(`${prefix}-symmetry-score`);
  const bar = document.getElementById(`${prefix}-symmetry-bar`);
  if (!el) return;
  if (score === null) { el.textContent = "—"; if (bar) bar.style.width = "0%"; return; }
  el.textContent  = `${score}%`;
  if (bar) bar.style.width = `${score}%`;
}

// ── YÜZ SECTION GÜNCELLEYİCİ ─────────────────────────────────────────────────
function updateFaceSection(prefix, faceLm, faceShapes) {
  const section = document.getElementById(`${prefix}-face-section`);
  if (!section) return;

  if (!faceLm) {
    section.style.display = "none";
    return;
  }
  section.style.display = "block";

  const roll  = calcHeadRoll(faceLm);
  const yaw   = calcHeadYaw(faceLm);
  const mouth = calcMouthOpen(faceLm);

  const rollVal  = document.getElementById(`${prefix}-val-roll`);
  const rollBar  = document.getElementById(`${prefix}-bar-roll`);
  const rollCard = document.getElementById(`${prefix}-card-roll`);
  if (rollVal && roll !== null) {
    rollVal.textContent = `${roll}°`;
    if (rollBar)  rollBar.style.width  = `${Math.min(100, (Math.abs(roll) / 90) * 100)}%`;
    if (rollCard) rollCard.classList.add("active");
  }

  const yawVal  = document.getElementById(`${prefix}-val-yaw`);
  const yawBar  = document.getElementById(`${prefix}-bar-yaw`);
  const yawCard = document.getElementById(`${prefix}-card-yaw`);
  if (yawVal && yaw !== null) {
    yawVal.textContent = `${yaw}°`;
    if (yawBar)  yawBar.style.width  = `${Math.min(100, (Math.abs(yaw) / 90) * 100)}%`;
    if (yawCard) yawCard.classList.add("active");
  }

  const mouthVal  = document.getElementById(`${prefix}-val-mouth`);
  const mouthBar  = document.getElementById(`${prefix}-bar-mouth`);
  const mouthCard = document.getElementById(`${prefix}-card-mouth`);
  if (mouthVal && mouth !== null) {
    mouthVal.textContent = `${mouth}%`;
    if (mouthBar)  mouthBar.style.width  = `${Math.min(100, mouth)}%`;
    if (mouthCard) mouthCard.classList.add("active");
  }

  const categories = faceShapes?.categories ?? [];
  const expr  = guessExpression(categories);
  const smile = (getBlendShape(categories, "mouthSmileLeft") + getBlendShape(categories, "mouthSmileRight")) / 2;
  const blink = (getBlendShape(categories, "eyeBlinkLeft")   + getBlendShape(categories, "eyeBlinkRight"))  / 2;

  const exprEl  = document.getElementById(`${prefix}-stat-expr`);
  const smileEl = document.getElementById(`${prefix}-stat-smile`);
  const blinkEl = document.getElementById(`${prefix}-stat-blink`);
  if (exprEl)  exprEl.textContent  = expr;
  if (smileEl) {
    smileEl.textContent = smile > 0.4 ? "EVET" : "HAYIR";
    smileEl.className   = "stat-value" + (smile > 0.4 ? " yes" : "");
  }
  if (blinkEl) {
    blinkEl.textContent = blink > 0.6 ? "EVET" : "HAYIR";
    blinkEl.className   = "stat-value" + (blink > 0.6 ? " yes" : "");
  }
}

// ── CANVAS AÇI ETİKETİ ───────────────────────────────────────────────────────
function drawAngleLabel(ctx, x, y, degrees, color) {
  if (degrees === null) return;
  const cx = (1 - x) * canvasElement.width;
  const cy = y       * canvasElement.height;
  ctx.save();
  ctx.font         = "bold 13px 'Space Mono', monospace";
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  const text = `${degrees}°`;
  const tw   = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(5,10,14,0.8)";
  ctx.beginPath();
  ctx.roundRect(cx - tw / 2, cy - 9, tw, 18, 4);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1;
  ctx.stroke();
  ctx.fillStyle   = color;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 6;
  ctx.fillText(text, cx, cy);
  ctx.restore();
}

// ── PANEL GÜNCELLEME ─────────────────────────────────────────────────────────
function syncPersonCards(activeSlots) {
  const count = activeSlots.length;
  personCountEl.textContent = count;
  noPersonMsg.style.display = count === 0 ? "flex" : "none";

  const activeSet = new Set(activeSlots);
  for (let j = 0; j < MAX_PERSONS; j++) {
    const existing = document.getElementById(`person-card-${j}`);
    if (activeSet.has(j)) {
      if (!existing) {
        const card = createPersonCard(j);
        let inserted = false;
        const allCards = metricsPanel.querySelectorAll(".person-card");
        for (const c of allCards) {
          if (parseInt(c.id.replace("person-card-", "")) > j) {
            metricsPanel.insertBefore(card, c);
            inserted = true;
            break;
          }
        }
        if (!inserted) metricsPanel.appendChild(card);
      }
    } else {
      if (existing) existing.remove();
    }
  }
}

// ── YÜZ → KİŞİ EŞLEŞTİRME ───────────────────────────────────────────────────
function matchFacesToSlots(faceLandmarks, poseLandmarks, poseToSlot) {
  const faceToSlot = new Array(faceLandmarks.length).fill(-1);
  const slotUsed   = new Set();

  for (let fi = 0; fi < faceLandmarks.length; fi++) {
    const faceNose = faceLandmarks[fi][1];
    if (!faceNose) continue;

    let minDist  = 0.3;
    let bestSlot = -1;

    for (let pi = 0; pi < poseLandmarks.length; pi++) {
      const slotIdx = poseToSlot[pi];
      if (slotIdx === -1 || slotUsed.has(slotIdx)) continue;

      const poseNose = poseLandmarks[pi][0];
      if (!poseNose) continue;

      const d = Math.hypot(faceNose.x - poseNose.x, faceNose.y - poseNose.y);
      if (d < minDist) {
        minDist  = d;
        bestSlot = slotIdx;
      }
    }

    if (bestSlot !== -1) {
      faceToSlot[fi] = bestSlot;
      slotUsed.add(bestSlot);
    }
  }

  return faceToSlot;
}

// ── WEBCAM TOGGLE ─────────────────────────────────────────────────────────────
webcamButton.addEventListener("click", async () => {
  if (!modelsReady) {
    alert("Modeller henüz yükleniyor, lütfen bekleyin.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    btnText.textContent = "KAMERAYI AÇ";
    webcamButton.classList.remove("active");
    screenshotBtn.disabled = true;
    video.srcObject?.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    cameraPlaceholder.classList.remove("hidden");
    trackerSlots.forEach(s => { s.active = false; s.age = 0; });
    syncPersonCards([]);
    fpsDisplay.textContent = "0";
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
      });
      webcamRunning = true;
      btnText.textContent = "KAMERAYI KAPAT";
      webcamButton.classList.add("active");
      screenshotBtn.disabled = false;
      cameraPlaceholder.classList.add("hidden");
      video.srcObject = stream;
      video.onloadeddata = () => {
        video.play().then(() => predictWebcam());
      };
    } catch (err) {
      console.error("Kamera erişim hatası:", err);
      alert("Kameraya erişilemiyor: " + err.message);
    }
  }
});

// ── SCREENSHOT ────────────────────────────────────────────────────────────────
screenshotBtn.addEventListener("click", () => {
  const tmp = document.createElement("canvas");
  tmp.width  = canvasElement.width;
  tmp.height = canvasElement.height;
  const tCtx = tmp.getContext("2d");
  tCtx.save();
  tCtx.translate(tmp.width, 0);
  tCtx.scale(-1, 1);
  tCtx.drawImage(video, 0, 0, tmp.width, tmp.height);
  tCtx.restore();
  tCtx.save();
  tCtx.translate(tmp.width, 0);
  tCtx.scale(-1, 1);
  tCtx.drawImage(canvasElement, 0, 0);
  tCtx.restore();
  const link = document.createElement("a");
  link.download = `hareket-analiz-${Date.now()}.png`;
  link.href = tmp.toDataURL("image/png");
  link.click();
});

// ── ANA DÖNGÜ ─────────────────────────────────────────────────────────────────
async function predictWebcam() {
  if (!webcamRunning) return;

  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width  = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  const now = performance.now();
  fpsFrameCount++;
  if (now - fpsLastTime >= 500) {
    fpsDisplay.textContent = Math.round(fpsFrameCount * 1000 / (now - fpsLastTime));
    fpsFrameCount = 0;
    fpsLastTime   = now;
  }

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    try {
      const poseResults = poseLandmarker.detectForVideo(video, now);
      const handResults = handLandmarker.detectForVideo(video, now);
      const faceResults = faceLandmarker.detectForVideo(video, now);

      const poseLandmarks   = poseResults.landmarks       ?? [];
      const faceLandmarks   = faceResults.faceLandmarks   ?? [];
      const faceBlendshapes = faceResults.faceBlendshapes ?? [];
      const poseCount       = poseLandmarks.length;

      const poseToSlot  = poseCount > 0 ? matchPosesToSlots(poseLandmarks) : [];
      const activeSlots = [...new Set(poseToSlot.filter(j => j !== -1))];
      syncPersonCards(activeSlots);

      const slotFaceLm = {};
      const slotFaceBs = {};

      if (faceLandmarks.length > 0 && poseLandmarks.length > 0) {
        const faceToSlot = matchFacesToSlots(faceLandmarks, poseLandmarks, poseToSlot);
        for (let fi = 0; fi < faceLandmarks.length; fi++) {
          const slot = faceToSlot[fi];
          if (slot !== -1) {
            slotFaceLm[slot] = faceLandmarks[fi];
            slotFaceBs[slot] = faceBlendshapes[fi];
          }
        }
      }

      activeSlots.forEach(j => {
        updateStat(`p${j}`, "left-hand",  false);
        updateStat(`p${j}`, "right-hand", false);
      });

      for (let i = 0; i < poseCount; i++) {
        const slotIdx = poseToSlot[i];
        if (slotIdx === -1) continue;

        const lm     = poseLandmarks[i];
        const color  = PERSON_COLORS[slotIdx % PERSON_COLORS.length];
        const prefix = `p${slotIdx}`;

        canvasCtx.shadowColor = color.hex;
        canvasCtx.shadowBlur  = 12;
        drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: color.hex, lineWidth: 3 });
        drawingUtils.drawLandmarks(lm, { color: "#ffffff", fillColor: color.hex, lineWidth: 1, radius: 4 });
        canvasCtx.shadowBlur = 0;

        const smx = (lm[11].x + lm[12].x) / 2;
        const smy = Math.min(lm[11].y, lm[12].y) - 0.06;
        const lx  = (1 - smx) * canvasElement.width;
        const ly  = smy * canvasElement.height;
        canvasCtx.save();
        canvasCtx.font          = "bold 14px 'Barlow Condensed', sans-serif";
        canvasCtx.textAlign     = "center";
        canvasCtx.textBaseline  = "middle";
        const lw = canvasCtx.measureText(color.name).width + 14;
        canvasCtx.fillStyle = "rgba(5,10,14,0.85)";
        canvasCtx.beginPath();
        canvasCtx.roundRect(lx - lw / 2, ly - 11, lw, 22, 4);
        canvasCtx.fill();
        canvasCtx.strokeStyle = color.hex;
        canvasCtx.lineWidth   = 1.5;
        canvasCtx.stroke();
        canvasCtx.fillStyle   = color.hex;
        canvasCtx.shadowColor = color.hex;
        canvasCtx.shadowBlur  = 8;
        canvasCtx.fillText(color.name, lx, ly);
        canvasCtx.restore();

        const leftElbow      = calcAngle(lm[11], lm[13], lm[15]);
        const rightElbow     = calcAngle(lm[12], lm[14], lm[16]);
        const leftShoulder   = calcAngle(lm[13], lm[11], lm[23]);
        const rightShoulder  = calcAngle(lm[14], lm[12], lm[24]);
        const leftKnee       = calcAngle(lm[23], lm[25], lm[27]);
        const rightKnee      = calcAngle(lm[24], lm[26], lm[28]);
        const shoulderSpread = calcAngle(lm[23], lm[11], lm[12]);
        const hipAngle       = calcAngle(lm[11], lm[23], lm[25]);

        updateAngleCard(prefix, "left-elbow",      leftElbow);
        updateAngleCard(prefix, "right-elbow",     rightElbow);
        updateAngleCard(prefix, "left-shoulder",   leftShoulder);
        updateAngleCard(prefix, "right-shoulder",  rightShoulder);
        updateAngleCard(prefix, "left-knee",       leftKnee);
        updateAngleCard(prefix, "right-knee",      rightKnee);
        updateAngleCard(prefix, "shoulder-spread", shoulderSpread);
        updateAngleCard(prefix, "hip",             hipAngle);

        if (leftElbow  !== null) drawAngleLabel(canvasCtx, lm[13].x, lm[13].y, leftElbow,  color.hex);
        if (rightElbow !== null) drawAngleLabel(canvasCtx, lm[14].x, lm[14].y, rightElbow, color.hex);
        if (leftKnee   !== null) drawAngleLabel(canvasCtx, lm[25].x, lm[25].y, leftKnee,   color.hex);
        if (rightKnee  !== null) drawAngleLabel(canvasCtx, lm[26].x, lm[26].y, rightKnee,  color.hex);

        const lmEl = document.getElementById(`${prefix}-stat-landmarks`);
        if (lmEl) lmEl.textContent = lm.length;

        const pairs = [
          [leftElbow,    rightElbow],
          [leftShoulder, rightShoulder],
          [leftKnee,     rightKnee]
        ].filter(([a, b]) => a !== null && b !== null);
        if (pairs.length > 0) {
          const avg = pairs.reduce((s, [a, b]) => s + Math.abs(a - b), 0) / pairs.length;
          updateSymmetry(prefix, Math.round(Math.max(0, 100 - (avg / 90) * 100)));
        } else {
          updateSymmetry(prefix, null);
        }

        const faceLm = slotFaceLm[slotIdx];
        const faceBs = slotFaceBs[slotIdx];

        if (faceLm) {
          canvasCtx.save();
          canvasCtx.globalAlpha = 0.85;
          drawingUtils.drawConnectors(faceLm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: color.hex, lineWidth: 1.5 });
          drawingUtils.drawConnectors(faceLm, FaceLandmarker.FACE_LANDMARKS_LIPS,      { color: color.hex, lineWidth: 1.2 });
          drawingUtils.drawConnectors(faceLm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,  { color: "#30FF30", lineWidth: 1.2 });
          drawingUtils.drawConnectors(faceLm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030", lineWidth: 1.2 });
          canvasCtx.globalAlpha = 1;
          canvasCtx.restore();
        }

        updateFaceSection(prefix, faceLm ?? null, faceBs ?? null);
      }

      const rawHands      = handResults.landmarks    ?? [];
      const rawHandedness = handResults.handednesses ?? handResults.handedness ?? [];

      for (let h = 0; h < rawHands.length; h++) {
        const handLandmarks = rawHands[h];
        const catArr = rawHandedness[h];
        let catName  = null;
        if (Array.isArray(catArr) && catArr.length > 0) catName = catArr[0].categoryName;
        else if (catArr?.categoryName) catName = catArr.categoryName;

        const isUserLeftHand  = (catName === "Right");
        const isUserRightHand = (catName === "Left");

        const wrist = handLandmarks[0];
        let minDist  = Infinity;
        let bestSlot = activeSlots.length > 0 ? activeSlots[0] : 0;

        for (let pi = 0; pi < poseCount; pi++) {
          const sIdx = poseToSlot[pi];
          if (sIdx === -1) continue;
          const lm = poseLandmarks[pi];
          const d  = Math.min(
            Math.hypot(wrist.x - lm[15].x, wrist.y - lm[15].y),
            Math.hypot(wrist.x - lm[16].x, wrist.y - lm[16].y)
          );
          if (d < minDist) { minDist = d; bestSlot = sIdx; }
        }

        if (isUserLeftHand)  updateStat(`p${bestSlot}`, "left-hand",  true);
        if (isUserRightHand) updateStat(`p${bestSlot}`, "right-hand", true);

        const hColor = PERSON_COLORS[bestSlot % PERSON_COLORS.length];
        canvasCtx.shadowColor = hColor.hex;
        canvasCtx.shadowBlur  = 10;
        drawingUtils.drawConnectors(handLandmarks, HandLandmarker.HAND_CONNECTIONS, { color: hColor.hex, lineWidth: 2.5 });
        drawingUtils.drawLandmarks(handLandmarks, { color: "#ffffff", fillColor: hColor.hex, lineWidth: 1, radius: 3 });
        canvasCtx.shadowBlur = 0;
      }

    } catch (err) {
      console.error("Tahmin hatası:", err);
    }

    canvasCtx.restore();
  }

  if (webcamRunning) requestAnimationFrame(predictWebcam);
}
