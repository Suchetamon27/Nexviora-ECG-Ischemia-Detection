#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <math.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Forward declarations for C++ compilation
static void updateOledSample(uint16_t ecgVal);
static void sensorPoll();
static void storeEcgSample();
static void initializeSensor();
static void initOledAndMpu();
static void pollMPU();
static void renderOledDisplay();
static void ecgSamplerTask(void *pvParameters);
void handleRoot();
void handleReading();
void handleRec();
void handleExport();
void handleScan();
void handleData();
static String runI2CScan();
static void resetReadingCounters();

// =====================================================
// NEXVIORA — ESP32-S3 ECG + MAX30102 DASHBOARD
// 50Hz notch + baseline filter, rail detection,
// ECG R-peak HR + PPG HR, gated reading,
// 60s recording w/ CSV export (ECG + PPG + beats)
// =====================================================

// ---------------- GPIO ASSIGNMENTS ----------------
static const int ECG_PIN  = 4;   // AD8232 OUT — ADC1 (works with WiFi on)
static const int LO_P_PIN = 5;   // AD8232 LO+ (HIGH = lead off)
static const int LO_M_PIN = 6;   // AD8232 LO- (HIGH = lead off)
static const int SDN_PIN  = 7;   // AD8232 SDN — ACTIVE LOW: HIGH = RUNNING, LOW = SHUTDOWN!

static const int I2C_SDA  = 8;   // MAX30102 SDA
static const int I2C_SCL  = 9;   // MAX30102 SCL
static const int MAX_INT  = 10;  // reserved, unused

// ---------------- I2C MUTEX & THREAD SAFETY ----------------
static SemaphoreHandle_t i2cMutex = NULL;

static inline bool lockI2C(uint32_t timeoutMs = 25) {
  if (!i2cMutex) return true;
  return (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(timeoutMs)) == pdTRUE);
}

static inline void unlockI2C() {
  if (i2cMutex) xSemaphoreGive(i2cMutex);
}

// ---------------- OLED DISPLAY (128x32 / 128x64) ----------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
static bool oledReady = false;

static uint8_t oledPoints[128];
static uint8_t sweepX = 0;
static float oledEcgMin = 1800.0f;
static float oledEcgMax = 2200.0f;

// ---------------- MPU-6500 / MPU-6050 ----------------
static bool mpuReady = false;
static float motionG = 1.0f;
static float pitchDeg = 0.0f;
static float rollDeg = 0.0f;
static bool isMoving = false;
static int16_t mpuAx = 0, mpuAy = 0, mpuAz = 0;


// ---------------- WIFI ----------------
const char* AP_SSID = "Nexviora-ECG";
const char* AP_PASS = "nexviora123";

WebServer server(80);

// ---------------- ECG SAMPLING ----------------
static const uint32_t SAMPLE_PERIOD_US = 4000;   // 250 Hz
static const uint32_t SAMPLE_PERIOD_MS = 4;
static const uint16_t BUFFER_SIZE = 512;         // power of two
static const uint32_t MAX_SAMPLES_PER_RESPONSE = 64;

struct EcgSample {
  uint32_t seq;
  uint16_t value;
};

static EcgSample ecgBuffer[BUFFER_SIZE];
static volatile uint32_t sampleSeq = 0;
static portMUX_TYPE ecgMux = portMUX_INITIALIZER_UNLOCKED;

static TaskHandle_t samplerTaskHandle = nullptr;
static bool samplerTaskRunning = false;
static uint32_t lastSampleUs = 0;
static uint32_t lastSensorLoopMs = 0;

// ---------------- PPG DISPLAY BUFFER ----------------
struct PpgSample {
  uint32_t seq;
  int16_t value;
};

static const uint16_t PPG_BUFFER_SIZE = 256;
static PpgSample ppgBuffer[PPG_BUFFER_SIZE];
static volatile uint32_t ppgSeq = 0;
static portMUX_TYPE ppgMux = portMUX_INITIALIZER_UNLOCKED;
static const uint32_t MAX_PPG_PER_RESPONSE = 32;

// ---------------- ECG FILTERING (50 Hz notch + baseline) ----------------
static float nf_x1 = 0, nf_x2 = 0, nf_y1 = 0, nf_y2 = 0;
static float ecgDc = 2048.0f;

static uint8_t railFlags[64];
static uint8_t railIdx = 0;

static uint16_t ecgFilter(int32_t raw) {
  railFlags[railIdx] = (raw <= 10 || raw >= 4085) ? 1 : 0;
  railIdx = (railIdx + 1) & 63;

  // Slow baseline tracker (~8 s) — gentle enough to preserve ST segment
  ecgDc += ((float)raw - ecgDc) * 0.0005f;

  float x = (float)raw - ecgDc;

  // RBJ biquad notch: fs=250, f0=50, Q=25
  float y = 0.981335f * x - 0.606517f * nf_x1 + 0.981335f * nf_x2
                           + 0.606517f * nf_y1 - 0.962678f * nf_y2;

  nf_x2 = nf_x1; nf_x1 = x;
  nf_y2 = nf_y1; nf_y1 = y;

  return (uint16_t)constrain(y + 2048.0f, 0.0f, 4095.0f);
}

static uint8_t railPercent() {
  uint8_t n = 0;
  for (uint8_t i = 0; i < 64; i++) n += railFlags[i];
  return (uint8_t)((n * 100) / 64);
}

// ---------------- ECG R-PEAK DETECTION ----------------
static float eEnvMax = 0, eEnvMin = 0;
static bool  eArmed = false;
static uint32_t eLastBeatMs = 0;
static uint16_t eIntervals[4];
static uint8_t  eIntCount = 0, eIntSpot = 0;
static float ecgBpmVal = 0;
static bool  ecgHrValid = false;
static uint32_t ecgBeatCount = 0;

static const float ECG_MIN_AMP = 45.0f;   // counts on filtered signal

static void ecgBeatDetect(uint16_t value, uint32_t nowMs) {
  float ac = (float)value - 2048.0f;

  eEnvMax = (ac > eEnvMax) ? ac : eEnvMax * 0.998f;
  eEnvMin = (ac < eEnvMin) ? ac : eEnvMin * 0.998f;
  float amp = eEnvMax - eEnvMin;

  if (ecgHrValid && (nowMs - eLastBeatMs) > 3000) {
    ecgHrValid = false;
    eIntCount = 0;
    eArmed = false;
  }

  // Refractory period: ignore events within 280ms of previous beat (>214 bpm)
  if (amp > ECG_MIN_AMP && (nowMs - eLastBeatMs) >= 280) {
    float trigHi = eEnvMin + amp * 0.65f;
    float trigLo = eEnvMin + amp * 0.25f;

    if (!eArmed) {
      if (ac < trigLo) eArmed = true;
    } else if (ac > trigHi) {
      eArmed = false;
      uint32_t interval = nowMs - eLastBeatMs;
      eLastBeatMs = nowMs;

      bool ok = (interval >= 300 && interval <= 2000);
      if (ok && eIntCount > 0) {
        float prev = eIntervals[(eIntSpot - 1) & 3];
        if (interval < prev * 0.6f || interval > prev * 1.6f) ok = false;
      }

      if (ok) {
        eIntervals[eIntSpot] = (uint16_t)interval;
        eIntSpot = (eIntSpot + 1) & 3;
        if (eIntCount < 4) eIntCount++;

        uint32_t sum = 0;
        for (uint8_t i = 0; i < eIntCount; i++) sum += eIntervals[i];
        ecgBpmVal = 60000.0f * eIntCount / (float)sum;
        ecgHrValid = true;
        ecgBeatCount++;
      } else {
        eIntCount = 0;
      }
    }
  } else if (amp <= ECG_MIN_AMP) {
    eArmed = false;
    eIntCount = 0;
  }
}

// =====================================================
// FINGER-READING GATE
// =====================================================

static volatile bool readingActive = false;
static uint32_t readingStartMs = 0;
static String readingStatus = "Counting OFF — press 'Start reading' before placing your finger.";

// =====================================================
// RECORDING (60 s capture -> CSV export)
// =====================================================

static const uint32_t REC_MAX_SECONDS = 60;
static const uint32_t REC_ECG_CAP = REC_MAX_SECONDS * 250;   // 15000
static const uint32_t REC_PPG_CAP = REC_MAX_SECONDS * 100;   // 6000
static const uint32_t REC_BEAT_CAP = 512;

static uint16_t *recEcg = nullptr;    // filtered ECG @ 250 Hz
static int16_t  *recPpg = nullptr;    // IR AC component @ 100 Hz
struct BeatEvent { uint32_t t; uint16_t bpm10; uint16_t spo210; };
static BeatEvent *recBeats = nullptr;

static volatile bool recording = false;
static volatile bool serialStreaming = true;
static volatile bool recFull = false;
static volatile uint32_t recEcgCount = 0;
static volatile uint32_t recPpgCount = 0;
static volatile uint32_t recBeatCount = 0;
static uint32_t recStartMs = 0;
static bool recAvailable = false;
static portMUX_TYPE recMux = portMUX_INITIALIZER_UNLOCKED;

// =====================================================
// MAX30102 — REGISTER-LEVEL DRIVER
// =====================================================

#define MAX_ADDR        0x57
#define REG_INT_STATUS1 0x00
#define REG_FIFO_WR     0x04
#define REG_FIFO_OVF    0x05
#define REG_FIFO_RD     0x06
#define REG_FIFO_DATA   0x07
#define REG_FIFO_CFG    0x08
#define REG_MODE_CFG    0x09
#define REG_SPO2_CFG    0x0A
#define REG_LED1_PA     0x0C
#define REG_LED2_PA     0x0D
#define REG_PART_ID     0xFF

static const uint8_t MODE_SPO2 = 0x03;

static const uint8_t LED_RED_CURRENT = 0x28;
static const uint8_t LED_IR_CURRENT  = 0x28;

static const uint8_t SPO2_CONFIG_VAL = (0x3 << 5) | (0x1 << 2) | 0x3;
static const uint8_t FIFO_CONFIG_VAL = 0x10;

// ---------------- SENSOR STATE ----------------
bool sensorReady = false;
uint8_t sensorPartID = 0;
String sensorDiag = "Not attempted yet";
String lastScanResult = "not run yet";

static uint32_t nextRetryMs = 0;
static const uint32_t RETRY_INTERVAL_MS = 5000;
static uint32_t initAttemptCount = 0;

// ---------------- OPTICAL SIGNAL PROCESSING ----------------
static float dcRed = 0, dcIr = 0;
static bool  dcValid = false;
static float smRed = 0, smIr = 0;
static float envIrMax = 0, envIrMin = 0;
static float envRedMax = 0, envRedMin = 0;
static bool  armed = false;
static uint32_t lastBeatMs = 0;
static uint16_t intervals[4];
static uint8_t  intervalCount = 0;
static uint8_t  intervalSpot = 0;
static float hrBpm = 0;
static float spo2Pct = 0;
static bool  hrValid = false;
static bool  spo2Valid = false;
static uint32_t lastAcceptedBeatMs = 0;

static const float MIN_PULSE_AMP = 35.0f;
static const uint32_t BEAT_TIMEOUT_MS = 3000;

static volatile float dbgDcIr = 0;
static volatile float dbgAmpIr = 0;
static volatile uint32_t dbgBeats = 0;
static volatile uint32_t dbgLastInterval = 0;

// =====================================================
// HTML DASHBOARD
// =====================================================

const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nexviora ECG Monitor</title>

<style>
* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 18px;
  background: #10151d;
  color: #edf2f7;
  font-family: system-ui, Arial, sans-serif;
}

main { max-width: 900px; margin: auto; }

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 4px;
}

h1 { font-size: 25px; margin: 0; }

.subtitle {
  color: #9aa8b8;
  font-size: 13px;
  margin: 5px 0 18px;
}

.card {
  background: #19212c;
  border: 1px solid #2a3543;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 14px;
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.metrics { display: flex; gap: 12px; flex-wrap: wrap; }

.metric { flex: 1; min-width: 150px; }

.muted { color: #9aa8b8; font-size: 12px; }

.big {
  font-size: 32px;
  font-weight: 650;
  margin-top: 5px;
  font-variant-numeric: tabular-nums;
}

.units { font-size: 13px; color: #9aa8b8; font-weight: 400; }

.pill {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 20px;
  background: #263342;
  color: #cbd5e1;
  font-size: 12px;
}

canvas {
  width: 100%;
  height: 260px;
  display: block;
  background: #0d131b;
  border-radius: 8px;
  margin-top: 12px;
}

canvas.small { height: 150px; }

.leadwarn {
  color: #fbbf24;
  font-weight: 600;
  font-size: 14px;
  margin: 10px 0 0;
}

.status { font-size: 14px; margin: 8px 0; }

.hint { color: #fbbf24; font-size: 13px; }

.notice {
  color: #b7c2cf;
  font-size: 12px;
  line-height: 1.55;
}

button {
  padding: 9px 14px;
  border: 1px solid #3b4959;
  background: #263342;
  color: #edf2f7;
  border-radius: 9px;
  cursor: pointer;
}

button:hover { background: #334357; }

button.active { background: #7f1d1d; border-color: #b91c1c; }

button:disabled { opacity: 0.45; cursor: default; }

.btnrow { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }

.zoom-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.zoom-btn {
  padding: 5px 12px;
  font-size: 11px;
  font-weight: 500;
  border-radius: 6px;
  background: #202b38;
  border: 1px solid #324356;
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.15s ease;
}
.zoom-btn:hover { background: #2c3c4e; color: #fff; }
.zoom-btn.active {
  background: #0284c7;
  border-color: #38bdf8;
  color: #ffffff;
  font-weight: 600;
  box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
}
.wave-legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 8px;
  font-size: 11px;
  color: #94a3b8;
  background: #121822;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid #202b38;
}
.wave-tag { font-weight: 600; }
.wave-p { color: #38bdf8; }
.wave-qrs { color: #f43f5e; }
.wave-t { color: #facc15; }

@media (max-width: 520px) {
  body { padding: 12px; }
  canvas { height: 220px; }
  canvas.small { height: 130px; }
  .big { font-size: 27px; }
  .card { padding: 13px; }
}
</style>
</head>

<body>
<main>

<header>
  <h1>Nexviora</h1>
  <span class="pill" id="connection">Connecting...</span>
</header>

<p class="subtitle">ECG and optical pulse prototype monitor</p>

<section class="card">
  <div class="row">
    <strong>ECG waveform</strong>
    <span class="muted">AD8232 · 250 Hz · 50 Hz notch on</span>
  </div>

  <div class="zoom-bar">
    <span class="muted" style="font-size:12px;">Scale / Window:</span>
    <button class="zoom-btn active" id="zoom1" onclick="setZoom(180, this)">🔍 1 Beat (~0.7s) [P-QRS-T Zoom]</button>
    <button class="zoom-btn" id="zoom2" onclick="setZoom(320, this)">2 Beats (~1.3s)</button>
    <button class="zoom-btn" id="zoom3" onclick="setZoom(500, this)">3 Beats (~2.0s)</button>
  </div>

  <canvas id="plot"></canvas>

  <div class="wave-legend">
    <span><span class="wave-tag wave-p">● P Wave</span> Atria</span>
    <span><span class="wave-tag wave-qrs">▲ QRS Spike</span> Ventricles</span>
    <span><span class="wave-tag wave-t">● T Wave</span> Repol</span>
    <span style="margin-left:auto; color:#64748b;">Adjust window to see individual waves</span>
  </div>

  <p class="leadwarn" id="leadBanner" style="display:none;">
    ⚠ Electrodes disconnected or signal saturated — quality gate active.
  </p>

  <p class="muted" id="leadStatus">Lead status: waiting...</p>
  <p class="muted" id="sampleStatus">Samples: 0</p>
</section>

<section class="card">
  <div class="row">
    <strong>Pulse signal (IR)</strong>
    <button id="readBtn" onclick="toggleReading()">▶ Start reading</button>
  </div>

  <canvas id="pplot" class="small"></canvas>

  <p class="status" id="readStatus">Counting OFF</p>
  <p class="muted" id="ppgDebug">Beats: -- · Amp: -- · DC: -- · Interval: --</p>
  <p class="hint" id="fingerHint"></p>
</section>

<section class="metrics">
  <div class="card metric">
    <div class="muted">ECG heart rate</div>
    <div class="big">
      <span id="ecgHr">--</span>
      <span class="units">bpm</span>
    </div>
  </div>

  <div class="card metric">
    <div class="muted">Optical heart rate</div>
    <div class="big">
      <span id="hr">--</span>
      <span class="units">bpm</span>
    </div>
  </div>

  <div class="card metric">
    <div class="muted">SpO₂ estimate</div>
    <div class="big">
      <span id="spo2">--</span>
      <span class="units">%</span>
    </div>
  </div>
</section>

<section class="card">
  <div class="row">
    <strong>Body motion &amp; tilt (MPU-6500)</strong>
    <span class="pill" id="motionStatus" style="background:#14532d; color:#86efac;">Stable</span>
  </div>
  <div style="display:flex; align-items:center; gap:16px; margin-top:12px;">
    <div style="width:68px; height:68px; border-radius:50%; background:#0d131b; border:1px solid #2a3543; position:relative; overflow:hidden; flex-shrink:0;">
      <div style="position:absolute; top:33px; left:0; width:100%; height:1px; background:#25313e;"></div>
      <div style="position:absolute; left:33px; top:0; width:1px; height:100%; background:#25313e;"></div>
      <div id="tiltBubble" style="width:14px; height:14px; border-radius:50%; background:#38bdf8; position:absolute; top:27px; left:27px; transition:transform 0.05s linear; box-shadow:0 0 8px rgba(56,189,248,0.5);"></div>
    </div>
    <div style="flex:1; font-size:13px; color:#cbd5e1; display:flex; flex-direction:column; gap:6px;">
      <div class="row"><span>Pitch (Tilt F/B):</span><strong id="pitchVal">0.0°</strong></div>
      <div class="row"><span>Roll (Tilt L/R):</span><strong id="rollVal">0.0°</strong></div>
      <div class="row"><span>Acceleration:</span><strong id="motionVal">1.00 g</strong></div>
    </div>
  </div>
</section>

<section class="card">
  <div class="row">
    <strong>Recording &amp; export</strong>
    <span class="muted">up to 60 s · CSV</span>
  </div>

  <div class="btnrow">
    <button id="recBtn" onclick="toggleRec()">● Record</button>
    <button id="exportBtn" onclick="location.href='/rec/export'" disabled>⬇ Download CSV</button>
  </div>

  <p class="muted" id="recStatus">Idle — press Record to capture ECG, pulse and beats.</p>
</section>

<section class="card">
  <div class="row">
    <div class="muted">Sensor status (live)</div>
    <button onclick="scanBus()">Run I2C scan</button>
  </div>

  <p class="status" id="sensorStatus">Waiting for ESP32...</p>
  <p class="muted" id="sensorID"></p>
  <p class="muted" id="scanResult">I2C scan: not run yet</p>

  <p class="notice">
    Prototype display only. Values may be inaccurate or unavailable due
    to motion, contact, signal noise, or sensor limitations. This system
    is not a medical device and must not be used to make health decisions.
  </p>
</section>

</main>

<script>
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const pcanvas = document.getElementById("pplot");
const pctx = pcanvas.getContext("2d");

const history = [];
const ecgScaleHistory = [];
const irHistory = [];
let maxEcgPoints = 180; // Default: Zoomed in to 1 single beat (~0.72s at 250Hz)!
const MAX_PPG_POINTS = 200;

let lastSeq = 0;
let lastPpgSeq = 0;
let totalReceived = 0;
let busy = false;

function setZoom(pts, btn) {
  maxEcgPoints = pts;
  document.querySelectorAll('.zoom-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  if (history.length > maxEcgPoints) {
    history.splice(0, history.length - maxEcgPoints);
  }
  draw();
}

function setupCanvas(c, cx) {
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();

  c.width = Math.max(1, Math.floor(rect.width * dpr));
  c.height = Math.max(1, Math.floor(rect.height * dpr));

  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawTrace(cx, c, data, color, maxPts, baselineRef) {
  const w = c.clientWidth;
  const h = c.clientHeight;

  if (!w || !h) return;

  cx.clearRect(0, 0, w, h);

  // Subtle minor medical grid lines (5 per major division)
  cx.strokeStyle = "#16202c";
  cx.lineWidth = 0.75;
  const minorX = w / 40;
  const minorY = h / 20;
  cx.beginPath();
  for (let x = minorX; x < w; x += minorX) {
    cx.moveTo(x, 0); cx.lineTo(x, h);
  }
  for (let y = minorY; y < h; y += minorY) {
    cx.moveTo(0, y); cx.lineTo(w, y);
  }
  cx.stroke();

  // Major medical grid lines
  cx.strokeStyle = "#253547";
  cx.lineWidth = 1.2;
  const majorX = w / 8;
  const majorY = h / 4;
  cx.beginPath();
  for (let x = majorX; x < w; x += majorX) {
    cx.moveTo(x, 0); cx.lineTo(x, h);
  }
  for (let y = majorY; y < h; y += majorY) {
    cx.moveTo(0, y); cx.lineTo(w, y);
  }
  cx.stroke();

  if (data.length < 2) return;

  // Compute scale bounds (uses baselineRef for ECG to prevent jumping, or self data for PPG dynamic auto-scale)
  const scaleRef = (baselineRef && baselineRef.length > 30) ? baselineRef : data;
  let min = scaleRef[0], max = scaleRef[0];
  for (let i = 0; i < scaleRef.length; i++) {
    const v = scaleRef[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  if (max - min < 20) {
    min -= 10;
    max += 10;
  }

  // 12% padding: gives maximum vertical expansion to see P-wave, Q-dip, R-spike, S-dip, T-wave
  const padding = (max - min) * 0.12;
  min -= padding;
  max += padding;

  cx.strokeStyle = color;
  cx.lineWidth = 2.2;
  cx.lineCap = "round";
  cx.lineJoin = "round";
  cx.beginPath();

  const total = maxPts || 500;
  data.forEach((value, index) => {
    const x = index * w / (total - 1);
    const y = h - ((value - min) / (max - min)) * h;

    if (index === 0) {
      cx.moveTo(x, y);
    } else {
      cx.lineTo(x, y);
    }
  });

  cx.stroke();
}

function draw() {
  drawTrace(ctx, canvas, history, "#4ade80", maxEcgPoints, ecgScaleHistory);
  drawTrace(pctx, pcanvas, irHistory, "#f59e0b", MAX_PPG_POINTS);
}

function resizeAll() {
  setupCanvas(canvas, ctx);
  setupCanvas(pcanvas, pctx);
  draw();
}

window.addEventListener("resize", resizeAll);

async function poll() {
  if (busy) return;
  busy = true;

  try {
    const response = await fetch(
      "/data?after=" + lastSeq + "&pafter=" + lastPpgSeq,
      { cache: "no-store" }
    );

    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }

    const data = await response.json();

    lastSeq = data.next;
    lastPpgSeq = data.pnext;

    for (const value of data.ecg) {
      history.push(value);
      ecgScaleHistory.push(value);
      totalReceived++;

      if (history.length > maxEcgPoints) {
        history.shift();
      }
      if (ecgScaleHistory.length > 500) {
        ecgScaleHistory.shift();
      }
    }

    for (const value of data.ir) {
      irHistory.push(value);

      if (irHistory.length > MAX_PPG_POINTS) {
        irHistory.shift();
      }
    }

    draw();

    document.getElementById("connection").textContent = "Connected";

    const railPct = data.railPct || 0;
    const leadOff = data.loP === 1 || data.loM === 1 || railPct > 25;
    document.getElementById("leadBanner").style.display =
      leadOff ? "block" : "none";

    document.getElementById("leadStatus").textContent =
      "LO+ = " + data.loP + " · LO- = " + data.loM +
      " · railed: " + railPct + "%" +
      (railPct > 25 ? " (Quality Gated)" : " (Clean)");

    document.getElementById("sampleStatus").textContent =
      "Samples received: " + totalReceived;

    document.getElementById("ecgHr").textContent =
      data.ecgHr > 0 ? data.ecgHr.toFixed(0) : "--";

    document.getElementById("hr").textContent =
      data.hr > 0 ? data.hr.toFixed(0) : "--";

    document.getElementById("spo2").textContent =
      data.spo2 > 0 ? data.spo2.toFixed(0) : "--";

    document.getElementById("sensorStatus").textContent =
      data.diag || "No diagnostic available";

    document.getElementById("sensorID").textContent =
      data.partID >= 0
        ? "PART_ID: 0x" + data.partID.toString(16).toUpperCase()
        : "";

    document.getElementById("ppgDebug").textContent =
      "Beats: " + data.beats +
      " · Amp: " + Math.round(data.ampIr) +
      " · DC: " + Math.round(data.dcIr) +
      " · Interval: " + (data.lastInt > 0 ? data.lastInt + " ms" : "--");

    const readBtn = document.getElementById("readBtn");
    readBtn.textContent = data.reading ? "■ Stop reading" : "▶ Start reading";
    readBtn.classList.toggle("active", data.reading);

    lastFingerDetected = (data.dcIr > 8000);
    const hint = document.getElementById("fingerHint");

    if (!data.pox) {
      hint.textContent = "";
    } else if (!data.reading) {
      hint.textContent =
        "Counting is OFF — press Start reading, then place your fingertip.";
    } else if (data.ampIr < 30) {
      hint.textContent =
        "No pulse signal — place fingertip FLAT over the sensor LEDs, press lightly, hold still.";
    } else if (data.beats === 0) {
      hint.textContent =
        "Signal detected — hold still, first reading in a few beats.";
    } else {
      hint.textContent = "";
    }

    if (data.pitch !== undefined) {
      const p = data.pitch;
      const r = data.roll;
      const m = data.motion || 1.0;
      document.getElementById("pitchVal").textContent = (p >= 0 ? "+" : "") + p.toFixed(1) + "°";
      document.getElementById("rollVal").textContent = (r >= 0 ? "+" : "") + r.toFixed(1) + "°";
      document.getElementById("motionVal").textContent = m.toFixed(2) + " g";
      const bx = Math.max(-20, Math.min(20, (r / 30) * 20));
      const by = Math.max(-20, Math.min(20, (p / 30) * 20));
      document.getElementById("tiltBubble").style.transform = `translate(${bx}px, ${by}px)`;
      const mStat = document.getElementById("motionStatus");
      if (data.isMoving) {
        mStat.textContent = "Moving";
        mStat.style.background = "#7f1d1d";
        mStat.style.color = "#fca5a5";
      } else {
        mStat.textContent = "Stable";
        mStat.style.background = "#14532d";
        mStat.style.color = "#86efac";
      }
    }

    const recBtn = document.getElementById("recBtn");
    const recStatus = document.getElementById("recStatus");
    const exportBtn = document.getElementById("exportBtn");

    recBtn.textContent = data.rec ? "■ Stop" : "● Record";
    recBtn.classList.toggle("active", data.rec);

    if (data.rec) {
      recStatus.textContent =
        "Recording… " + data.recSec + " s / 60 s — " +
        data.recEcg + " ECG, " + data.recPpg + " PPG samples";
    } else if (data.recFull) {
      recStatus.textContent =
        "Buffer full (60 s) — " + data.recEcg + " ECG, " +
        data.recPpg + " PPG, " + data.recBeats + " beats. Download below.";
    } else if (data.recEcg > 0) {
      recStatus.textContent =
        "Captured " + data.recSec + " s — " + data.recEcg +
        " ECG, " + data.recPpg + " PPG, " + data.recBeats +
        " beats. Download below.";
    } else {
      recStatus.textContent = "Idle — press Record to capture ECG, pulse and beats.";
    }

    exportBtn.disabled = data.recEcg === 0;

  } catch (error) {
    document.getElementById("connection").textContent = "Disconnected";
  } finally {
    busy = false;
    setTimeout(poll, 70);
  }
}

async function toggleReading() {
  try {
    const r = await fetch("/reading", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "on=toggle"
    });
    await r.json();
  } catch (e) {}
}

async function toggleRec() {
  try {
    const r = await fetch("/rec", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "on=toggle"
    });
    await r.json();
  } catch (e) {}
}

async function scanBus() {
  const el = document.getElementById("scanResult");
  el.textContent = "I2C scan: running (may take a few seconds)...";

  try {
    const r = await fetch("/i2cscan", { cache: "no-store" });
    const d = await r.json();
    el.textContent = "I2C scan: " + d.scan;
  } catch (e) {
    el.textContent = "I2C scan: request failed";
  }
}

resizeAll();
setTimeout(poll, 80);
</script>
</body>
</html>
)rawliteral";


// =====================================================
// ECG RING BUFFER + RECORDING CAPTURE
// =====================================================

static void storeEcgSample() {
  int32_t raw = (int32_t)analogRead(ECG_PIN);
  uint16_t value = ecgFilter(raw);

  ecgBeatDetect(value, millis());
  updateOledSample(value);

  portENTER_CRITICAL(&ecgMux);
  uint32_t seq = sampleSeq;
  EcgSample &slot = ecgBuffer[seq & (BUFFER_SIZE - 1u)];
  slot.seq = seq;
  slot.value = value;
  sampleSeq = seq + 1;
  portEXIT_CRITICAL(&ecgMux);

  bool shouldStream = serialStreaming;
  portENTER_CRITICAL(&recMux);
  if (recording && recAvailable && recEcg && recEcgCount < REC_ECG_CAP) {
    recEcg[recEcgCount++] = value;
    if (recEcgCount >= REC_ECG_CAP) {
      recFull = true;
    }
  }
  portEXIT_CRITICAL(&recMux);

  if (shouldStream) {
    Serial.printf("REC,%u,%u,%u,%.1f,%.1f,%.1f,%.1f,%.2f\n", millis(), value, (uint32_t)dbgDcIr, ecgBpmVal, spo2Pct, pitchDeg, rollDeg, motionG);
  }
}

static inline void publishPpg(float v) {
  int16_t s = (int16_t)constrain(v, -32768.0f, 32767.0f);

  portENTER_CRITICAL(&ppgMux);
  uint32_t seq = ppgSeq;
  PpgSample &slot = ppgBuffer[seq & (PPG_BUFFER_SIZE - 1u)];
  slot.seq = seq;
  slot.value = s;
  ppgSeq = seq + 1;
  portEXIT_CRITICAL(&ppgMux);

  portENTER_CRITICAL(&recMux);
  if (recording && recPpgCount < REC_PPG_CAP) {
    recPpg[recPpgCount++] = s;
    if (recPpgCount >= REC_PPG_CAP && recEcgCount >= REC_ECG_CAP) {
      recording = false;
      recFull = true;
    }
  }
  portEXIT_CRITICAL(&recMux);
}

// =====================================================
// MAX30102 LOW-LEVEL I2C
// =====================================================

static bool maxWrite8(uint8_t reg, uint8_t val) {
  if (!lockI2C(25)) return false;
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(reg);
  Wire.write(val);
  uint8_t err = Wire.endTransmission();
  unlockI2C();
  return err == 0;
}

static bool maxRead8(uint8_t reg, uint8_t &val) {
  if (!lockI2C(25)) return false;
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) { unlockI2C(); return false; }
  if (Wire.requestFrom((uint8_t)MAX_ADDR, (uint8_t)1) != 1) { unlockI2C(); return false; }
  val = Wire.read();
  unlockI2C();
  return true;
}

// =====================================================
// OPTICAL SIGNAL PROCESSING (100 Hz)
// =====================================================

static void processOpticalSample(int32_t red, int32_t ir) {
  uint32_t now = millis();

  if (!dcValid) {
    dcRed = red;
    dcIr = ir;
    smRed = smIr = 0;
    envIrMax = envIrMin = envRedMax = envRedMin = 0;
    dcValid = true;
    return;
  }

  dcRed += (red - dcRed) / 256.0f;
  dcIr  += (ir  - dcIr ) / 256.0f;

  float acRed = red - dcRed;
  float acIr  = ir  - dcIr;
  smRed += (acRed - smRed) * 0.2f;
  smIr  += (acIr  - smIr ) * 0.2f;

  publishPpg(smIr);

  envIrMax  = (smIr  > envIrMax ) ? smIr  : envIrMax  * 0.997f;
  envIrMin  = (smIr  < envIrMin ) ? smIr  : envIrMin  * 0.997f;
  envRedMax = (smRed > envRedMax) ? smRed : envRedMax * 0.997f;
  envRedMin = (smRed < envRedMin) ? smRed : envRedMin * 0.997f;

  float ampIr  = envIrMax  - envIrMin;
  float ampRed = envRedMax - envRedMin;

  dbgDcIr  = dcIr;
  dbgAmpIr = ampIr;

  // Automatic finger detection: open air ~500, under finger > 8,000
  bool fingerDetected = (dcIr > 8000.0f);
  if (!fingerDetected) {
    readingActive = false;
    hrValid = false;
    spo2Valid = false;
    armed = false;
    intervalCount = 0;
    hrBpm = 0;
    spo2Pct = 0;
    return;
  }
  readingActive = true;

  if (hrValid && (now - lastAcceptedBeatMs) > BEAT_TIMEOUT_MS) {
    hrValid = false;
    spo2Valid = false;
    intervalCount = 0;
    armed = false;
  }

  if (ampIr > MIN_PULSE_AMP && ampIr > dcIr * 0.002f) {
    float trigHigh = envIrMin + ampIr * 0.60f;
    float trigLow  = envIrMin + ampIr * 0.25f;

    if (!armed) {
      if (smIr < trigLow) armed = true;
    } else if (smIr > trigHigh) {
      armed = false;
      uint32_t interval = now - lastBeatMs;
      lastBeatMs = now;

      bool ok = (interval >= 300 && interval <= 1500);

      if (ok && intervalCount > 0) {
        float prev = intervals[(intervalSpot - 1) & 3];
        if (interval < prev * 0.6f || interval > prev * 1.5f) ok = false;
      }

      if (ok) {
        intervals[intervalSpot] = (uint16_t)interval;
        intervalSpot = (intervalSpot + 1) & 3;
        if (intervalCount < 4) intervalCount++;

        uint32_t sum = 0;
        for (uint8_t i = 0; i < intervalCount; i++) sum += intervals[i];
        hrBpm = 60000.0f * intervalCount / (float)sum;
        hrValid = true;
        lastAcceptedBeatMs = now;

        dbgBeats++;
        dbgLastInterval = interval;

        if (ampRed > 15.0f && ampIr > 15.0f && dcRed > 500.0f && dcIr > 500.0f) {
          float R = (ampRed / dcRed) / (ampIr / dcIr);
          float s = 110.0f - 25.0f * R;
          s = constrain(s, 88.0f, 100.0f);
          spo2Pct = spo2Valid ? (spo2Pct * 0.75f + s * 0.25f) : s;
          spo2Valid = true;
        }

        portENTER_CRITICAL(&recMux);
        if (recording && recBeats && recBeatCount < REC_BEAT_CAP) {
          recBeats[recBeatCount].t     = now - recStartMs;
          recBeats[recBeatCount].bpm10 = (uint16_t)(hrBpm * 10.0f);
          recBeats[recBeatCount].spo210 = spo2Valid ? (uint16_t)(spo2Pct * 10.0f) : 0;
          recBeatCount++;
        }
        portEXIT_CRITICAL(&recMux);
      } else {
        intervalCount = 0;
      }
    }
  } else {
    armed = false;
    intervalCount = 0;
  }
}

// =====================================================
// MAX30102 FIFO POLLING + INIT
// =====================================================


// =====================================================
// MPU-6500 & PURE GRAPH OLED DRIVERS
// =====================================================

void initOledAndMpu() {
  if (!i2cMutex) {
    i2cMutex = xSemaphoreCreateMutex();
  }

  if (lockI2C(100)) {
    // 1. OLED Init
    if (!oledReady) {
      if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        oledReady = true;
        display.clearDisplay();
        display.display();
      }
    }

    // 2. MPU-6500 / 6050 Init (Handles WHO_AM_I = 0x70 or 0x68)
    if (!mpuReady) {
      Wire.beginTransmission(0x68);
      Wire.write(0x75); // WHO_AM_I
      if (Wire.endTransmission(false) == 0 && Wire.requestFrom((uint8_t)0x68, (uint8_t)1) == 1) {
        uint8_t who = Wire.read();
        if (who == 0x70 || who == 0x68 || who == 0x71 || who == 0x73) {
          Wire.beginTransmission(0x68);
          Wire.write(0x6B); // PWR_MGMT_1
          Wire.write(0x00); // Wake up
          Wire.endTransmission();
          mpuReady = true;
        }
      }
    }
    unlockI2C();
  }
}

void pollMPU() {
  if (!mpuReady) return;
  if (!lockI2C(10)) return;

  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) == 0 && Wire.requestFrom((uint8_t)0x68, (uint8_t)6) == 6) {
    mpuAx = (int16_t)(Wire.read() << 8 | Wire.read());
    mpuAy = (int16_t)(Wire.read() << 8 | Wire.read());
    mpuAz = (int16_t)(Wire.read() << 8 | Wire.read());
    float ax = (float)mpuAx / 16384.0f;
    float ay = (float)mpuAy / 16384.0f;
    float az = (float)mpuAz / 16384.0f;
    motionG = sqrt(ax * ax + ay * ay + az * az);
    isMoving = (fabs(motionG - 1.0f) > 0.18f);

    float p = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0f / M_PI;
    float r = atan2(ay, az) * 180.0f / M_PI;
    pitchDeg = pitchDeg * 0.7f + p * 0.3f;
    rollDeg  = rollDeg * 0.7f + r * 0.3f;
  }
  unlockI2C();
}

void updateOledSample(uint16_t ecgVal) {
  // Real-time running envelope tracker for dynamic auto-centering & full-scale height
  if (ecgVal < oledEcgMin) oledEcgMin = ecgVal;
  else oledEcgMin += 0.8f;

  if (ecgVal > oledEcgMax) oledEcgMax = ecgVal;
  else oledEcgMax -= 0.8f;

  float span = oledEcgMax - oledEcgMin;
  if (span < 160.0f) span = 160.0f; // minimum amplitude baseline

  // FULL VERTICAL HEIGHT: y = 0 (top) to 31 (bottom) with 1px margin
  float norm = ((float)ecgVal - oledEcgMin) / span;
  uint8_t y = (uint8_t)(31.0f - constrain(norm, 0.0f, 1.0f) * 30.0f);

  // Subsample by 2: advance sweepX every 2nd sample (125 Hz)
  // At 125 Hz, 128 pixels = 1.024 seconds, perfectly fitting 1-2 complete heartbeats on the screen!
  static uint8_t sampleDiv = 0;
  sampleDiv = !sampleDiv;
  if (sampleDiv) {
    oledPoints[sweepX] = y;
    sweepX = (sweepX + 1) % 128;
  }
}

void renderOledDisplay() {
  if (!oledReady) return;
  static uint32_t lastOledFrame = 0;
  if (millis() - lastOledFrame < 33) return; // 30 FPS refresh rate
  lastOledFrame = millis();

  if (!lockI2C(15)) return;

  display.clearDisplay();

  // PURE DYNAMIC FULL-HEIGHT CARDIAC OSCILLOSCOPE WAVEFORM (ZERO TEXT CLUTTER)
  for (int x = 0; x < 127; x++) {
    // 4-pixel erase gap ahead of sweep head for classic medical monitor sweep
    if ((x >= sweepX && x <= sweepX + 4) || (sweepX >= 124 && x <= (sweepX + 4) % 128)) {
      continue;
    }
    int xNext = x + 1;
    display.drawLine(x, oledPoints[x], xNext, oledPoints[xNext], SSD1306_WHITE);
  }

  display.display();
  unlockI2C();
}

static void sensorPoll() {
  uint8_t wr, rd;

  if (!maxRead8(REG_FIFO_WR, wr) || !maxRead8(REG_FIFO_RD, rd)) {
    return;
  }

  uint8_t avail = (uint8_t)(wr - rd) & 0x1F;
  if (avail > 8) avail = 8;

  for (uint8_t i = 0; i < avail; i++) {
    if (!lockI2C(20)) return;
    Wire.beginTransmission(MAX_ADDR);
    Wire.write(REG_FIFO_DATA);
    if (Wire.endTransmission(false) != 0) { unlockI2C(); return; }

    if (Wire.requestFrom((uint8_t)MAX_ADDR, (uint8_t)6) != 6) { unlockI2C(); return; }

    uint8_t r0 = Wire.read(), r1 = Wire.read(), r2 = Wire.read();
    uint8_t i0 = Wire.read(), i1 = Wire.read(), i2 = Wire.read();
    unlockI2C();

    uint32_t red = (((uint32_t)r0 << 16) | ((uint32_t)r1 << 8) | r2) & 0x3FFFF;
    uint32_t ir  = (((uint32_t)i0 << 16) | ((uint32_t)i1 << 8) | i2) & 0x3FFFF;

    processOpticalSample((int32_t)red, (int32_t)ir);
  }
}

String runI2CScan() {
  String out = "";
  uint8_t found = 0;

  for (uint8_t addr = 1; addr < 127; ++addr) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      if (found) out += " ";
      out += "0x";
      out += String(addr, HEX);
      found++;
    }
  }

  if (found == 0) out = "no devices responded";

  lastScanResult = out;
  return out;
}

bool readPartID(uint8_t &partID) {
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(REG_PART_ID);

  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MAX_ADDR, (uint8_t)1) != 1) return false;

  partID = Wire.read();
  return true;
}

static void resetReadingCounters() {
  hrValid = false;
  spo2Valid = false;
  intervalCount = 0;
  armed = false;
  dbgBeats = 0;
  dbgLastInterval = 0;
  lastAcceptedBeatMs = 0;
}

void initializeSensor() {
  if (sensorReady) return; // Prevent resetting running sensor and flickering LEDs
  sensorDiag = "Checking sensor...";

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);   // Fast I2C mode reduces bus transaction time to <1ms
  Wire.setTimeOut(50);
  delay(100);

  Wire.beginTransmission(MAX_ADDR);
  uint8_t error = Wire.endTransmission();

  if (error != 0) {
    if (initAttemptCount < 2) runI2CScan();

    sensorDiag = "Nothing answered on the bus (err " + String(error) +
                 ", scan: " + lastScanResult + ").";
    initAttemptCount++;
    sensorReady = false;
    return;
  }

  if (!readPartID(sensorPartID)) {
    sensorDiag = "0x57 answered but PART_ID read failed - unstable bus.";
    initAttemptCount++;
    sensorReady = false;
    return;
  }

  if (sensorPartID == 0x00 || sensorPartID == 0xFF) {
    sensorDiag = "PART_ID is garbage: marginal I2C bus.";
    initAttemptCount++;
    sensorReady = false;
    return;
  }

  if (sensorPartID == 0x11) {
    sensorDiag = "Chip is a MAX30100 (0x11) - this build drives a MAX30102.";
    initAttemptCount++;
    sensorReady = false;
    return;
  }

  if (sensorPartID != 0x15) {
    sensorDiag = "Unknown chip, PART_ID 0x" + String(sensorPartID, HEX) + ".";
    initAttemptCount++;
    sensorReady = false;
    return;
  }

  bool ok = true;
  ok &= maxWrite8(REG_MODE_CFG, 0x40);
  delay(10);

  ok &= maxWrite8(REG_FIFO_CFG,  FIFO_CONFIG_VAL);
  ok &= maxWrite8(REG_SPO2_CFG,  SPO2_CONFIG_VAL);
  ok &= maxWrite8(REG_LED1_PA,   LED_RED_CURRENT);
  ok &= maxWrite8(REG_LED2_PA,   LED_IR_CURRENT);
  ok &= maxWrite8(REG_FIFO_WR,   0x00);
  ok &= maxWrite8(REG_FIFO_OVF,  0x00);
  ok &= maxWrite8(REG_FIFO_RD,   0x00);
  ok &= maxWrite8(REG_MODE_CFG,  MODE_SPO2);

  if (!ok) {
    sensorDiag = "MAX30102 detected but register writes failed.";
    initAttemptCount++;
    sensorReady = false;
    return;
  }

  dcValid = false;
  resetReadingCounters();

  initAttemptCount = 0;
  sensorReady = true;
  sensorDiag = "MAX30102 ready. Amber trace is live; press 'Start reading' to count.";
}

// =====================================================
// READING GATE + RECORDING CONTROL
// =====================================================

void setReading(bool on, const String &why) {
  readingActive = on;

  if (on) {
    resetReadingCounters();
    readingStartMs = millis();
    readingStatus = "Counting ON — place fingertip on the sensor and hold still.";
  } else {
    readingStatus = why;
  }
}

void stopReadingAuto() {
  uint32_t now = millis();

  if (!readingActive) return;

  if (dbgBeats == 0 && (now - readingStartMs) > 12000) {
    setReading(false, "No pulse detected in 12 s — counting stopped. Check finger placement and press Start again.");
  } else if (dbgBeats > 0 && lastAcceptedBeatMs != 0 &&
             (now - lastAcceptedBeatMs) > 8000) {
    setReading(false, "Signal lost — counting auto-stopped. Press Start to go again.");
  }
}

// =====================================================
// HTTP HANDLERS
// =====================================================

void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleReading() {
  String cmd = server.hasArg("on") ? server.arg("on") : "";

  if (cmd == "toggle") {
    setReading(!readingActive, "Counting OFF — press 'Start reading' before placing your finger.");
  } else if (cmd == "1") {
    setReading(true, "");
  } else if (cmd == "0") {
    setReading(false, "Counting OFF — press 'Start reading' before placing your finger.");
  }

  String payload = "{\"reading\":";
  payload += readingActive ? "true" : "false";
  payload += "}";

  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", payload);
}

void handleRec() {
  String cmd = server.hasArg("on") ? server.arg("on") : "";

  bool start = false;

  if (cmd == "toggle") {
    start = !recording;
  } else if (cmd == "1") {
    start = true;
  } else if (cmd == "0") {
    start = false;
  }

  if (start && !recording && recAvailable) {
    portENTER_CRITICAL(&recMux);
    recEcgCount = 0;
    recPpgCount = 0;
    recBeatCount = 0;
    recFull = false;
    recStartMs = millis();
    recording = true;
    serialStreaming = true;
    portEXIT_CRITICAL(&recMux);
    Serial.println("#REC_START");
  } else if (!start && recording) {
    portENTER_CRITICAL(&recMux);
    recording = false;
    serialStreaming = false;
    portEXIT_CRITICAL(&recMux);
    Serial.println("#REC_STOP");
  }

  String payload = "{\"rec\":";
  payload += recording ? "true" : "false";
  payload += "}";

  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", payload);
}

void handleExport() {
  if (!recAvailable || (recEcgCount == 0 && recPpgCount == 0)) {
    server.send(404, "text/plain", "No recording available. Press Record first.");
    return;
  }

  uint32_t nEcg, nPpg, nBeats;
  portENTER_CRITICAL(&recMux);
  nEcg = recEcgCount;
  nPpg = recPpgCount;
  nBeats = recBeatCount;
  portEXIT_CRITICAL(&recMux);

  server.sendHeader("Content-Disposition",
                    "attachment; filename=nexviora_recording.csv");
  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, "text/csv", "");

  server.sendContent("type,t_ms,value1,value2\n");
  server.sendContent("# ECG : 250 Hz, filtered ADC (50Hz notch). value2 unused\n");
  server.sendContent("# PPG : 100 Hz, IR AC component. value2 unused\n");
  server.sendContent("# BEAT: value1 = HR bpm (PPG), value2 = SpO2 % (0 = n/a)\n");

  const uint32_t ROWS_PER_CHUNK = 64;
  String chunk;
  chunk.reserve(ROWS_PER_CHUNK * 32 + 16);

  for (uint32_t base = 0; base < nEcg; base += ROWS_PER_CHUNK) {
    chunk = "";
    uint32_t end = (base + ROWS_PER_CHUNK < nEcg) ? base + ROWS_PER_CHUNK : nEcg;

    for (uint32_t i = base; i < end; i++) {
      uint16_t v;
      portENTER_CRITICAL(&recMux);
      v = recEcg[i];
      portEXIT_CRITICAL(&recMux);

      chunk += "ECG,";
      chunk += String(i * 4);
      chunk += ",";
      chunk += String(v);
      chunk += ",0\n";
    }
    server.sendContent(chunk);
  }

  for (uint32_t base = 0; base < nPpg; base += ROWS_PER_CHUNK) {
    chunk = "";
    uint32_t end = (base + ROWS_PER_CHUNK < nPpg) ? base + ROWS_PER_CHUNK : nPpg;

    for (uint32_t i = base; i < end; i++) {
      int16_t v;
      portENTER_CRITICAL(&recMux);
      v = recPpg[i];
      portEXIT_CRITICAL(&recMux);

      chunk += "PPG,";
      chunk += String(i * 10);
      chunk += ",";
      chunk += String(v);
      chunk += ",0\n";
    }
    server.sendContent(chunk);
  }

  for (uint32_t base = 0; base < nBeats; base += ROWS_PER_CHUNK) {
    chunk = "";
    uint32_t end = (base + ROWS_PER_CHUNK < nBeats) ? base + ROWS_PER_CHUNK : nBeats;

    for (uint32_t i = base; i < end; i++) {
      uint32_t t;
      uint16_t b10, s10;
      portENTER_CRITICAL(&recMux);
      t   = recBeats[i].t;
      b10 = recBeats[i].bpm10;
      s10 = recBeats[i].spo210;
      portEXIT_CRITICAL(&recMux);

      chunk += "BEAT,";
      chunk += String(t);
      chunk += ",";
      chunk += String(b10 / 10.0f, 1);
      chunk += ",";
      chunk += (s10 == 0) ? String(0) : String(s10 / 10.0f, 1);
      chunk += "\n";
    }
    server.sendContent(chunk);
  }

  server.sendContent("");
}

void handleScan() {
  String payload = "{\"scan\":\"";

  if (sensorReady) {
    payload += "Sensor initialized - bus is working, scan skipped.";
  } else {
    Wire.setClock(100000);
    payload += runI2CScan();
  }

  payload += "\"}";
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", payload);
}

void handleData() {
  uint32_t after = 0;
  uint32_t pafter = 0;

  if (server.hasArg("after")) {
    after = (uint32_t)strtoul(server.arg("after").c_str(), nullptr, 10);
  }
  if (server.hasArg("pafter")) {
    pafter = (uint32_t)strtoul(server.arg("pafter").c_str(), nullptr, 10);
  }

  uint32_t current, pcurrent;
  portENTER_CRITICAL(&ecgMux);
  current = sampleSeq;
  portEXIT_CRITICAL(&ecgMux);
  portENTER_CRITICAL(&ppgMux);
  pcurrent = ppgSeq;
  portEXIT_CRITICAL(&ppgMux);

  uint32_t first = after;
  if (current - first > MAX_SAMPLES_PER_RESPONSE) {
    first = current - MAX_SAMPLES_PER_RESPONSE;
  }

  uint32_t pfirst = pafter;
  if (pcurrent - pfirst > MAX_PPG_PER_RESPONSE) {
    pfirst = pcurrent - MAX_PPG_PER_RESPONSE;
  }

  String json;
  json.reserve(1700);

  json = "{\"next\":";
  json += String(current);
  json += ",\"ecg\":[";

  bool comma = false;

  for (uint32_t seq = first; seq < current; seq++) {
    EcgSample s;

    portENTER_CRITICAL(&ecgMux);
    s = ecgBuffer[seq & (BUFFER_SIZE - 1u)];
    portEXIT_CRITICAL(&ecgMux);

    if (s.seq != seq) continue;

    if (comma) json += ",";
    json += String(s.value);
    comma = true;
  }

  json += "],\"ir\":[";
  comma = false;

  for (uint32_t seq = pfirst; seq < pcurrent; seq++) {
    PpgSample p;

    portENTER_CRITICAL(&ppgMux);
    p = ppgBuffer[seq & (PPG_BUFFER_SIZE - 1u)];
    portEXIT_CRITICAL(&ppgMux);

    if (p.seq != seq) continue;

    if (comma) json += ",";
    json += String(p.value);
    comma = true;
  }

  json += "],\"pnext\":";
  json += String(pcurrent);

  float hr   = hrValid   ? hrBpm   : 0.0f;
  float spo2 = spo2Valid ? spo2Pct : 0.0f;
  if (!isfinite(hr)) hr = 0;
  if (!isfinite(spo2)) spo2 = 0;

  int loP = digitalRead(LO_P_PIN);
  int loM = digitalRead(LO_M_PIN);
  uint8_t rPct = railPercent();

  // Signal quality gating: suppress noisy artifact HR if railed or lead-off
  bool ecgClean = ecgHrValid && (rPct <= 25) && (loP == 0) && (loM == 0);

  json += ",\"loP\":";
  json += String(loP);
  json += ",\"loM\":";
  json += String(loM);
  json += ",\"railPct\":";
  json += String(rPct);
  json += ",\"ecgHr\":";
  json += ecgClean ? String(ecgBpmVal, 1) : String(0);
  json += ",\"hr\":";
  json += String(hr, 1);
  json += ",\"spo2\":";
  json += String(spo2, 1);
  json += ",\"pox\":";
  json += sensorReady ? "true" : "false";
  json += ",\"partID\":";
  json += sensorReady ? String(sensorPartID) : String(-1);

  json += ",\"beats\":";
  json += String(dbgBeats);
  json += ",\"ampIr\":";
  json += String(dbgAmpIr, 0);
  json += ",\"dcIr\":";
  json += String(dbgDcIr, 0);
  json += ",\"lastInt\":";
  json += String(dbgLastInterval);

  json += ",\"reading\":";
  json += readingActive ? "true" : "false";

  uint32_t recSec = recording ? (millis() - recStartMs) / 1000 : (recEcgCount + 249) / 250;

  json += ",\"rec\":";
  json += recording ? "true" : "false";
  json += ",\"recFull\":";
  json += recFull ? "true" : "false";
  json += ",\"recSec\":";
  json += String(recSec);
  json += ",\"recEcg\":";
  json += String(recEcgCount);
  json += ",\"recPpg\":";
  json += String(recPpgCount);
  json += ",\"recBeats\":";
  json += String(recBeatCount);

  json += ",\"pitch\":"; json += String(pitchDeg, 1); json += ",\"roll\":"; json += String(rollDeg, 1); json += ",\"motion\":"; json += String(motionG, 2); json += ",\"isMoving\":"; json += isMoving ? "true" : "false";
  json += ",\"diag\":\"";
  json += sensorDiag;
  json += "\"}";

  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", json);
}

// =====================================================
// SAMPLER TASK (core 1, above loop priority)
// =====================================================

static void ecgSamplerTask(void *parameter) {
  (void)parameter;
  TickType_t lastWake = xTaskGetTickCount();
  uint32_t lastSensorMs = 0;

  for (;;) {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));

    storeEcgSample();

    uint32_t nowMs = millis();
    if (sensorReady && (nowMs - lastSensorMs) >= 10) {
      lastSensorMs = nowMs;
      sensorPoll();
    }
  }
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(LO_P_PIN, INPUT);
  pinMode(LO_M_PIN, INPUT);

  // AD8232 SDN is ACTIVE LOW:
  // Driving HIGH enables normal operation. (Driving LOW forces shutdown!)
  pinMode(SDN_PIN, OUTPUT);
  digitalWrite(SDN_PIN, HIGH);

  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  pinMode(MAX_INT, INPUT_PULLUP);

  for (uint16_t i = 0; i < BUFFER_SIZE; ++i) {
    ecgBuffer[i].seq = 0xFFFFFFFFu;
  }
  for (uint16_t i = 0; i < PPG_BUFFER_SIZE; ++i) {
    ppgBuffer[i].seq = 0xFFFFFFFFu;
  }
  memset(railFlags, 0, sizeof(railFlags));

  recEcg   = (uint16_t*)malloc(REC_ECG_CAP * sizeof(uint16_t));
  recPpg   = (int16_t*) malloc(REC_PPG_CAP * sizeof(int16_t));
  recBeats = (BeatEvent*)malloc(REC_BEAT_CAP * sizeof(BeatEvent));
  recAvailable = (recEcg && recPpg && recBeats);

  if (!recAvailable) {
    free(recEcg); free(recPpg); free(recBeats);
    recEcg = nullptr; recPpg = nullptr; recBeats = nullptr;
  }

  initializeSensor();
  initOledAndMpu();
  nextRetryMs = millis() + RETRY_INTERVAL_MS;

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.println("[WiFi] SoftAP active: SSID='Nexviora-ECG', IP=192.168.4.1");

  const char* STA_SSID1 = "OPPO K13x 5g s3cc";
  const char* STA_SSID2 = "OPPO K13x 5G s3cc";
  const char* STA_PASS  = "debargha";

  Serial.printf("[WiFi] Attempting connection to hotspot '%s'...\n", STA_SSID1);
  WiFi.begin(STA_SSID1, STA_PASS);

  uint32_t staWaitStart = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - staWaitStart < 5000)) {
    delay(200);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[WiFi] Retrying hotspot '%s'...\n", STA_SSID2);
    WiFi.begin(STA_SSID2, STA_PASS);
    staWaitStart = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - staWaitStart < 5000)) {
      delay(200);
      Serial.print(".");
    }
    Serial.println();
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] Connected to hotspot! Local IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[WiFi] Hotspot not connected yet (will auto-reconnect in background). AP active at http://192.168.4.1");
  }

  if (MDNS.begin("nexviora-ecg")) {
    Serial.println("[mDNS] Responder started: http://nexviora-ecg.local");
  }

  server.on("/", HTTP_GET, handleRoot);
  server.on("/data", HTTP_GET, handleData);
  server.on("/i2cscan", HTTP_GET, handleScan);
  server.on("/reading", HTTP_POST, handleReading);
  server.on("/rec", HTTP_POST, handleRec);
  server.on("/rec/export", HTTP_GET, handleExport);
  server.begin();

  if (xTaskCreatePinnedToCore(
        ecgSamplerTask,
        "ecg_sampler",
        4096,
        nullptr,
        3,
        &samplerTaskHandle,
        1
      ) == pdPASS) {
    samplerTaskRunning = true;
  } else {
    lastSampleUs = micros();
  }
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {
  server.handleClient();

  while (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "#REC_START") {
      portENTER_CRITICAL(&recMux);
      serialStreaming = true;
      recording = true;
      recEcgCount = 0;
      recPpgCount = 0;
      recBeatCount = 0;
      recFull = false;
      recStartMs = millis();
      portEXIT_CRITICAL(&recMux);
      Serial.println("#REC_STARTED");
    } else if (cmd == "#REC_STOP") {
      portENTER_CRITICAL(&recMux);
      serialStreaming = false;
      recording = false;
      portEXIT_CRITICAL(&recMux);
      Serial.println("#REC_STOPPED");
    }
  }

  if (!sensorReady) {
    if ((int32_t)(millis() - nextRetryMs) >= 0) {
      nextRetryMs = millis() + RETRY_INTERVAL_MS;
      initializeSensor();
      initOledAndMpu();
    }
  }

  // Non-blocking Wi-Fi auto-reconnect every 10 seconds if disconnected
  static uint32_t lastWifiCheck = 0;
  if (millis() - lastWifiCheck > 10000) {
    lastWifiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.begin("OPPO K13x 5g s3cc", "debargha");
    }
  }

  uint32_t now = millis();
  static uint32_t lastMpuTime = 0;
  if (now - lastMpuTime >= 50) {
    lastMpuTime = now;
    pollMPU();
  }

  renderOledDisplay();

  if (!samplerTaskRunning) {
    uint32_t nowMs = millis();

    if (sensorReady && (nowMs - lastSensorLoopMs) >= 10) {
      lastSensorLoopMs = nowMs;
      sensorPoll();
    }

    uint32_t now = micros();
    uint32_t missed = (uint32_t)(now - lastSampleUs) / SAMPLE_PERIOD_US;

    if (missed > 0) {
      uint32_t take = missed > 8 ? 8 : missed;
      lastSampleUs += missed * SAMPLE_PERIOD_US;

      for (uint32_t i = 0; i < take; ++i) {
        storeEcgSample();
      }
    }
  }
}
