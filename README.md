# Nexviora: Real-Time ECG Clinical Telemetry & Ischemia Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![ModernBERT](https://img.shields.io/badge/Backbone-ModernBERT%208192-black?style=for-the-badge&logo=huggingface)](https://huggingface.co/convaiinnovations/laya)
[![Laya Model](https://img.shields.io/badge/Local%20Model-Laya%20Multilingual%20(RL--Agent)-101010?style=for-the-badge)](https://huggingface.co/convaiinnovations/laya)
[![NVIDIA NIM](https://img.shields.io/badge/Diagnostic%20LLM-meta%2Fmuse--glimmer--30b-76B900?style=for-the-badge&logo=nvidia)](https://build.nvidia.com)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3%20%2B%20AD8232%20%2B%20MPU6050-E7352C?style=for-the-badge&logo=espressif)](https://espressif.com)
[![License](https://img.shields.io/badge/License-Research%20Prototype-lightgrey?style=for-the-badge)](#license)

**A high-precision biomedical telemetry platform uniting zero-phase DSP, local real-time reinforcement-learning decision gating, and frontier LLM electrophysiological synthesis for acute myocardial ischemia screening.**

[Architecture](#system-architecture) • [Laya Local Model](#local-ai-model-laya-multilingual) • [Diagnostic LLM](#diagnostic-llm-metamuse-glimmer-30b) • [Hardware & Calibration](#hardware-wiring--calibration) • [Quickstart](#quickstart--installation) • [Web Dashboards](#dashboards--user-interfaces)

</div>

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │    BIOMEDICAL HARDWARE SENSING FRONT-END    │
                    │  AD8232 ECG (1100× Gain) + MPU6050 6-Axis IMU │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      ESP32-S3 FIRMWARE (focused-tesla.ino)   │
                    │  • Biquad 50Hz Notch Filter + Baseline Drift  │
                    │  • Wi-Fi STA / HTTP Telemetry Streaming (250Hz)│
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼ (Wi-Fi / Serial Stream)
                    ┌──────────────────────────────────────────────┐
                    │          FASTAPI BACKEND DSP PIPELINE        │
                    │  • Bidirectional Zero-Phase Butterworth      │
                    │    Bandpass (0.5 – 35.0 Hz filtfilt)         │
                    │  • Dual-Median ST-Preserving Baseline Filter │
                    │  • Dynamic MPU Motion Gating (|dev| > 0.18g) │
                    └───────┬──────────────────────────────┬───────┘
                            │                              │
                            ▼                              ▼
 ┌───────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
 │   TIER 1: LOCAL LAYA MODEL (PORT 8085)    │  │ TIER 2: NVIDIA CLOUD LLM (PORT 8000)    │
 │   convaiinnovations/laya/multilingual     │  │ meta/muse-glimmer-30b                   │
 │   • ModernBERT (8,192 Tokens Context)     │  │ • Long-Context Multi-Beat Synthesis     │
 │   • 2-Layer RL-Agent Decision Head        │  │ • Calibrated Electrophysiology Analysis │
 │   • Sub-50ms CPU Pass 1 Quality Gating    │  │ • Four Ischemia Parameters Extraction   │
 │   • Evaluates SQI & MPU Discard Rules     │  │ • Standardized Clinical Reporting       │
 └───────────────────────────────────────────┘  └─────────────────────────────────────────┘
                            │                              │
                            └──────────────┬───────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      HIGH-CONTRAST MONOCHROME INTERFACES     │
                    │  • Live Telemetry Canvas Dashboard (:8000)   │
                    │  • Standalone Laya Local AI Studio (:8085)   │
                    └──────────────────────────────────────────────┘
```

---

## Two-Tiered AI Architecture

| Metric / Dimension | Tier 1: Local Laya Model | Tier 2: Diagnostic Frontier LLM |
|---|---|---|
| **Model ID** | `convaiinnovations/laya` (`multilingual`) | `meta/muse-glimmer-30b` |
| **Model Family** | ModernBERT (22 Layers) + RL Decision Head | Dense Autoregressive Medical-Reasoning LLM |
| **Context Length** | **8,192 Tokens** | **32,768+ Tokens** |
| **Execution Location** | **Local (CPU / GPU)** via `laya` Python SDK | **NVIDIA NIM API Cloud Endpoint** |
| **Inference Latency** | **~20 – 50 ms** (Sub-second real-time) | **~3 – 8 seconds** (Deep multi-step reasoning) |
| **Trigger Cycle** | Continuous (Every 3s / 5s chunk) | On-demand / Session Accumulation Trigger |
| **Primary Task** | Signal Quality Gate, MPU Motion Discard, Pass 1 Filter | ST-Segment Deviation, T-Wave/Q-Wave Pathology, Final Clinical Report |
| **Output Type** | Calibrated Probabilities (`noul`) & Decisions (`choice`) | Calibrated Clinical Markdown & JSON Diagnostic Report |

---

## Local AI Model: Laya Multilingual

Nexviora integrates **Laya** as its local, zero-shot structured evaluation engine. Unlike conventional LLMs that require high GPU memory and generate conversational hallucinations, Laya evaluates structured schemas directly in a single forward pass.

### Model Specifications
- **Hugging Face Hub ID**: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
- **Subfolder / Revision**: `multilingual` (`main`)
- **Local Cache Path**: `~/.cache/huggingface/hub/models--convaiinnovations--laya/`
- **Total Weights Size**: **~615 MB** (`model.safetensors`)
- **Backbone Architecture**:
  - `jhu-clsp/mmBERT-base` (ModernBERT architecture)
  - 22 Hidden Layers, 12 Attention Heads, 768 Hidden Dimension
  - Alternating **128-token sliding-window local attention** and global attention
  - Native **8,192 Position Embeddings** (`max_position_embeddings: 8192`)
- **Decision Head**:
  - 2-Layer Reinforcement Learning Decision Head (`rl-agent`) in `bfloat16`
  - Cost-weighted action escalation and calibrated confidence scaling

### Output Types & Schema
Laya evaluates custom Python dictionaries or JSON telemetry against two native schema types:
1. **`noul`**: Direct scalar probability estimation ($P(\text{Yes}) \in [0.0, 1.0]$) with calibrated confidence score without token-by-token text generation.
2. **`choice`**: Multi-class categorization across discrete clinical criteria with full probability distribution across all choices.

### Local Python Usage
```python
import laya

# Initialize the router preloading the multilingual 8k model
router = laya.Router(preload=["multilingual"])

# Input telemetry state
state = {
    "heart_rate_bpm": 84,
    "st_elevation_mm": 1.92,
    "motion_deviation_g": 0.03,
    "pr_baseline_adc": 2048
}

# Structured questions schema
questions = {
    "ischemia_risk": {
        "type": "noul",
        "instructions": "Estimate probability of acute myocardial ischemia."
    },
    "classification": {
        "type": "choice",
        "instructions": "Classify the segment into clinical tier.",
        "criteria": {
            "NORMAL": "Normal sinus rhythm, clean physiological tracing.",
            "WARNING": "Borderline ST-T deviations present.",
            "CRITICAL": "Significant ST elevation or pathological T-wave inversion."
        }
    }
}

result = router.predict(state, questions, model="multilingual", max_len=8192)
print("Ischemia Probability:", result["answers"]["ischemia_risk"]["noul"])
print("Triage Choice:", result["answers"]["classification"]["choice"])
```

### Automatic Model Preloader
We provide a dedicated verification and preloading script in the repository:
```bash
python scripts/setup_laya_model.py
```
This script downloads the weights if not cached, validates the integrity of the ModernBERT backbone, and executes a sub-50ms test forward pass.

---

## Diagnostic LLM: `meta/muse-glimmer-30b`

For deep clinical synthesis, all validated chunks accumulated during a monitoring session are dispatched to NVIDIA's NIM API hosting `meta/muse-glimmer-30b`.

### Calibrated Directives
- **Zero Hallucination Directive**: The model is provided with hardware calibration parameters and is instructed to never state that ST/T waves "cannot be assessed" or require visual grid paper.
- **Single-Column Telemetry Stream**: Telemetry samples are transmitted as raw, single-column floating-point ECG values with 20ms deltas, maximizing token density.
- **Four Research Ischemia Parameters**:
  1. **ST-Segment Deviation**: Evaluated at $J + 60\text{ms}$ relative to PR isoelectric baseline. Quantified in millivolts (mV) and millimeters (mm).
  2. **T-Wave Abnormality**: Assesses polarity (upright, inverted, biphasic), symmetry, and relative amplitude.
  3. **Q-Wave Abnormality**: Distinguishes non-pathological Q-waves from pathological necrosis markers (depth $>25\%$ of R-wave or duration $>40\text{ms}$).
  4. **ST-T Consistency**: Verifies morphological reproducibility across consecutive cycles in the multi-beat recording.

---

## Hardware Wiring & Calibration

### 1. Analog Front-End Specifications
- **Instrumentation Amplifier**: AD8232 with bandpass circuitry (Gain = **1100×**).
- **ADC Resolution**: ESP32-S3 12-bit SAR ADC ($0 - 4095$ counts, $V_{\text{ref}} = 3.3\text{V}$).
- **Electrophysiological Scale Factor**:
  $$\text{Scale} = \frac{3.3\,\text{V}}{4095\,\text{counts}} \times \frac{1000\,\text{mV/V}}{1100\,\text{Gain}} \approx 0.0007325\,\text{mV / count}$$
  $$\mathbf{1.0\,\text{mV}} \approx \mathbf{1365\,\text{ADC counts}} \quad \vert \quad \mathbf{0.1\,\text{mV (1.0 mm)}} \approx \mathbf{136.5\,\text{ADC counts}}$$
- **PR Isoelectric Baseline**: Centered at **~2048 ADC counts** (1.65V mid-supply virtual ground).

### 2. Pinout Connections
| Sensor / Peripheral | Sensor Pin | ESP32-S3 Pin | Function / Description |
|---|---|---|---|
| **AD8232 ECG** | `OUTPUT` | `GPIO 34 / ADC1_CH6` | Analog ECG waveform input |
| **AD8232 ECG** | `LO+` | `GPIO 18` | Leads-off positive detection |
| **AD8232 ECG** | `LO-` | `GPIO 19` | Leads-off negative detection |
| **AD8232 ECG** | `3.3V` / `GND` | `3.3V` / `GND` | Low-noise regulated power |
| **MPU6050 IMU** | `SDA` | `GPIO 21` | I2C Data (accelerometer & gyro) |
| **MPU6050 IMU** | `SCL` | `GPIO 22` | I2C Clock |
| **MAX30102 PPG** | `SDA` / `SCL` | `GPIO 21` / `GPIO 22` | Shared I2C bus for SpO2 / Pulse |

---

## Dashboards & User Interfaces

The platform features two distinct web interfaces styled in a high-contrast, black-and-white, shadowless design:

### 1. Main Telemetry Dashboard (`http://localhost:8000`)
- **Live Canvas Oscilloscope**: High-frequency real-time ECG charting with zero phase-lag.
- **Vitals HUD**: Displays Heart Rate (BPM), Blood Oxygen (SpO2), Finger Placement, and MPU Motion Deviation.
- **Telemetry Accumulator**: Shows session length and accumulated valid chunks ready for long-context analysis.
- **Session Controls**:
  - `[🚀 Send to LLM]`: Submits the complete multi-beat session to `meta/muse-glimmer-30b`.
  - `[📥 Download CSV]`: Exports the active recording as a calibrated CSV file.
  - `[📄 Download AI Report]`: Exports the formatted clinical diagnosis.

### 2. Laya Local AI Studio (`http://localhost:8085`)
- **Interactive Playground** for running single-pass inferences with `laya-multilingual`.
- **Preloaded Clinical Presets**:
  - ⚡ **ECG Ischemia & ST-T Anomaly Detection** (Nexviora production schema)
  - 🛡️ **ECG Signal Quality & Motion Gate** (MPU acceleration deviation, SQI score)
  - 💓 **Cardiac Rhythm & Arrhythmia** (Sinus rhythm, Tachycardia, Bradycardia, Atrial Fibrillation)
  - 🏥 **Emergency Department Medical Triage** (ESI Acuity tiers 1–5)
  - 🩺 **Hemodynamic Stability & Shock Index** (Shock index calculation, ICU monitoring)
  - 📩 **Customer Support Intent & Escalation**
  - ✏️ **Custom Blank Schema**
- **Hardware Integration**: Includes a **`[⚡ Sync Live Telemetry]`** button that pulls live hardware data directly from the running backend.

---

## Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Suchetamon27/Nexviora-ECG-Ischemia-Detection.git
cd Nexviora-ECG-Ischemia-Detection

# Create virtual environment
python3 -m venv backend/.venv
source backend/.venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Preload the Laya Model
Verify and cache the 615 MB ModernBERT Laya multilingual model locally:
```bash
python scripts/setup_laya_model.py
```

### 3. Flash ESP32 Firmware
1. Open `focused-tesla.ino` in Arduino IDE or configure with `arduino-cli`.
2. Enter your Wi-Fi credentials:
   ```cpp
   WiFi.mode(WIFI_STA);
   WiFi.begin("YOUR_WIFI_SSID", "YOUR_WIFI_PASSWORD");
   ```
3. Compile and upload:
   ```bash
   arduino-cli compile --fqbn espressif:esp32:esp32s3 focused-tesla.ino
   arduino-cli upload -p /dev/ttyACM0 --fqbn espressif:esp32:esp32s3 focused-tesla.ino
   ```

### 4. Configure NVIDIA API Key (Optional for LLM Synthesis)
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### 5. Launch Servers
**Terminal 1: Main Backend & Telemetry Server**
```bash
source backend/.venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*Access dashboard at: **http://localhost:8000***

**Terminal 2: Laya Local AI Studio (Optional)**
```bash
source backend/.venv/bin/activate
python laya_web_app.py
```
*Access studio at: **http://localhost:8085***

---

## API Endpoints Reference

| Endpoint | Method | Functionality |
|---|---|---|
| `/health` | `GET` | Reports connection status, IP, recording status, and buffer chunk count |
| `/ws/live` | `WebSocket` | Real-time 250 Hz waveform stream and telemetry broadcasting |
| `/api/stream/start` | `POST` | Initializes continuous recording session |
| `/api/stream/stop` | `POST` | Finalizes and closes active telemetry CSV file |
| `/api/llm/analyze` | `POST` | Packages all buffered session chunks and calls `meta/muse-glimmer-30b` |
| `/api/llm/clear` | `POST` | Flushes session buffer chunks |
| `/api/session/export-full-csv` | `GET` | Streams full-resolution continuous CSV export |
| `/api/config/wifi` | `POST` | Updates target ESP32 IP address dynamically without restart |
| `/api/predict` *(Port 8085)* | `POST` | Dispatches structured schema prediction to local Laya model |
| `/api/fetch-live-telemetry` *(Port 8085)* | `GET` | Fetches active hardware telemetry from backend on Port 8000 |

---

## License & Disclaimer
This software and hardware specification is a **Research Prototype** created for biomedical telemetry and computer-aided diagnostics research. It is not FDA/CE approved as a standalone medical diagnostic system.

---

| Made by Team - Nexviora for Hackspire hackathon 2026.
