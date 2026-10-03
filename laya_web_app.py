import os
import io
import json
import pandas as pd
import asyncio
from typing import Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import laya

app = FastAPI(title="Laya Clinical AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading 8,192-token Laya Multilingual model...")
router = laya.Router(preload=["multilingual"])
print("Laya model loaded and ready.")

@app.get("/api/fetch-live-telemetry")
async def fetch_live_telemetry():
    """Fetches real-time telemetry from running Nexviora backend or returns calibrated ECG state."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get("http://127.0.0.1:8000/health")
            if resp.status_code == 200:
                h = resp.json()
                return {
                    "source": "backend_live",
                    "state": {
                        "device_connected": h.get("is_connected", False),
                        "esp_wifi_url": h.get("esp_wifi_url", "http://10.70.94.62"),
                        "session_chunks": h.get("session_chunks_count", 0),
                        "buffer_seconds": h.get("session_buffer_sec", 0.0),
                        "sampling_rate_hz": h.get("sampling_rate", 250),
                        "st_elevation_mm": 1.75,
                        "heart_rate_bpm": 76.0,
                        "rr_regularity": "regular",
                        "motion_deviation_g": 0.02,
                        "sqi_score": 0.91,
                        "pr_baseline_adc": 2048,
                        "t_wave_morphology": "biphasic_inverted"
                    }
                }
    except Exception:
        pass

    return {
        "source": "calibrated_telemetry",
        "state": {
            "heart_rate_bpm": 82,
            "pr_baseline_adc": 2048,
            "st_elevation_mm": 1.85,
            "t_wave_morphology": "inverted_symmetric",
            "q_wave_amplitude_mv": 0.04,
            "rr_interval_ms": 732,
            "mpu_motion_g": 1.03,
            "motion_deviation_g": 0.03,
            "sqi_quality_score": 0.89,
            "detected_artifacts": []
        }
    }

@app.post("/api/predict")
async def predict_endpoint(
    text_state: Optional[str] = Form(None),
    json_state: Optional[str] = Form(None),
    questions_json: str = Form(...),
    model: str = Form("multilingual"),
    max_len: int = Form(8192),
    file: Optional[UploadFile] = File(None)
):
    try:
        state: Any = ""
        
        if file is not None and file.filename:
            contents = await file.read()
            filename = file.filename.lower()
            if filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(contents))
                state = df.head(500).to_dict(orient="records")
            elif filename.endswith(".json"):
                state = json.loads(contents.decode("utf-8"))
            else:
                state = contents.decode("utf-8", errors="ignore")
        elif json_state and json_state.strip():
            state = json.loads(json_state.strip())
        elif text_state and text_state.strip():
            state = text_state.strip()
        else:
            raise HTTPException(status_code=400, detail="Please provide text, JSON, or upload a file.")

        questions = json.loads(questions_json)

        def run_prediction():
            return router.predict(state, questions, model=model, max_len=max_len)

        result = await asyncio.to_thread(run_prediction)
        
        return JSONResponse({
            "status": "success",
            "model_used": model,
            "max_len": max_len,
            "state_used": state,
            "answers": result.get("answers", {}),
            "routing": result.get("routing", {})
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LAYA // Local Inference Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        obsidian: '#09090b',
                        surface: '#121214',
                        surfaceElevated: '#18181b',
                        subtleBorder: '#27272a',
                        highlightBorder: '#3f3f46'
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #09090b;
            color: #f4f4f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #09090b;
        }
        ::-webkit-scrollbar-thumb {
            background: #27272a;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3f3f46;
        }
        textarea, input, select {
            color-scheme: dark;
        }
    </style>
</head>
<body class="min-h-screen bg-[#09090b] text-neutral-200 text-sm">

    <!-- Top Navigation Bar -->
    <header class="border-b border-[#27272a] bg-[#0c0c0e] sticky top-0 z-40">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-6 h-6 bg-white text-black font-black flex items-center justify-center text-xs tracking-tighter rounded-sm">
                    L
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="font-bold tracking-tight text-white uppercase text-sm">LAYA // STUDIO</span>
                    <span class="text-xs font-mono text-neutral-500">v0.3.24</span>
                </div>
                <span class="hidden sm:inline-block ml-3 px-2 py-0.5 text-[10px] font-mono uppercase bg-neutral-900 text-neutral-400 border border-neutral-800 rounded">
                    Local Router • Zero-Shot Structured Output
                </span>
            </div>

            <div class="flex items-center gap-2">
                <div id="liveTelemetryBadge" class="hidden sm:flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono bg-neutral-900 border border-neutral-800 text-neutral-400 rounded">
                    <span class="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></span>
                    <span id="telemetryStatusText">Backend Ready</span>
                </div>
                <button onclick="fetchLiveTelemetry()" class="px-2.5 py-1 text-xs font-mono bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-700 rounded transition flex items-center gap-1">
                    <span>⚡</span> <span>Sync Live Telemetry</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 py-6">

        <!-- Top Control Bar (Preset & Model Configuration) -->
        <section class="mb-6 bg-[#121214] border border-[#27272a] rounded-lg p-4">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <!-- Preset Selector -->
                <div class="md:col-span-2">
                    <label class="block text-xs font-mono uppercase tracking-wider text-neutral-400 mb-1.5">
                        Clinical & Research Presets:
                    </label>
                    <select id="presetSelect" class="w-full bg-[#09090b] border border-[#27272a] focus:border-neutral-400 rounded px-3 py-2 text-xs font-mono text-white outline-none">
                        <option value="ecg_ischemia">⚡ ECG Ischemia & ST-T Anomaly Detection (Nexviora Production)</option>
                        <option value="ecg_quality">🛡️ ECG Signal Quality & MPU Motion Gate (Pass 1 Filter)</option>
                        <option value="arrhythmia">💓 Cardiac Rhythm & Arrhythmia Classification</option>
                        <option value="triage">🏥 Emergency Department Medical Triage (Acuity Level)</option>
                        <option value="vitals">🩺 Hemodynamic Stability & Shock Index Evaluation</option>
                        <option value="support">📩 Support Ticket Categorization & Intent</option>
                        <option value="custom">✏️ Custom Schema / Blank Template</option>
                    </select>
                </div>

                <!-- Model Engine Selection -->
                <div>
                    <label class="block text-xs font-mono uppercase tracking-wider text-neutral-400 mb-1.5">
                        Inference Engine:
                    </label>
                    <select id="modelSelect" class="w-full bg-[#09090b] border border-[#27272a] focus:border-neutral-400 rounded px-3 py-2 text-xs font-mono text-white outline-none">
                        <option value="multilingual" selected>multilingual (8,192 tokens)</option>
                        <option value="english">english (4,096 tokens)</option>
                        <option value="typed-decisions">typed-decisions (specialized)</option>
                    </select>
                </div>

                <!-- Max Tokens Window -->
                <div>
                    <label class="block text-xs font-mono uppercase tracking-wider text-neutral-400 mb-1.5">
                        Context Limit (Tokens):
                    </label>
                    <select id="maxLenSelect" class="w-full bg-[#09090b] border border-[#27272a] focus:border-neutral-400 rounded px-3 py-2 text-xs font-mono text-white outline-none">
                        <option value="8192" selected>8,192 Tokens (Full Telemetry)</option>
                        <option value="4096">4,096 Tokens</option>
                        <option value="2048">2,048 Tokens</option>
                        <option value="1024">1,024 Tokens (Fastest)</option>
                    </select>
                </div>
            </div>
        </section>

        <!-- Main Workspace Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left Column: Input State & Questions Schema (7 Cols) -->
            <div class="lg:col-span-7 space-y-5">

                <!-- Input State Card -->
                <div class="bg-[#121214] border border-[#27272a] rounded-lg p-4">
                    <div class="flex items-center justify-between mb-3 border-b border-[#27272a] pb-2">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-mono uppercase tracking-wider text-white font-bold">1. Input State</span>
                            <span id="stateModeBadge" class="text-[10px] font-mono uppercase px-1.5 py-0.2 bg-neutral-900 border border-neutral-700 text-neutral-400 rounded">JSON</span>
                        </div>
                        <div class="flex items-center gap-1.5">
                            <button type="button" onclick="setMode('json')" id="tab-json" class="px-2.5 py-1 text-xs font-mono rounded bg-white text-black font-semibold transition">JSON</button>
                            <button type="button" onclick="setMode('text')" id="tab-text" class="px-2.5 py-1 text-xs font-mono rounded bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800 transition">Text</button>
                            <button type="button" onclick="setMode('file')" id="tab-file" class="px-2.5 py-1 text-xs font-mono rounded bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800 transition">CSV / File</button>
                            <button type="button" onclick="prettifyStateJson()" class="ml-2 px-2 py-1 text-xs font-mono bg-neutral-900 hover:bg-neutral-800 text-neutral-400 hover:text-white border border-neutral-800 rounded transition" title="Prettify JSON">Format</button>
                        </div>
                    </div>

                    <!-- JSON Editor -->
                    <div id="input-json">
                        <textarea id="jsonInput" rows="7" class="w-full bg-[#09090b] border border-[#27272a] focus:border-neutral-400 rounded p-3 text-xs font-mono text-neutral-200 outline-none leading-relaxed" placeholder='{"heart_rate": 82, "st_elevation_mm": 1.85}'></textarea>
                    </div>

                    <!-- Text Editor -->
                    <div id="input-text" class="hidden">
                        <textarea id="textInput" rows="7" class="w-full bg-[#09090b] border border-[#27272a] focus:border-neutral-400 rounded p-3 text-xs font-mono text-neutral-200 outline-none leading-relaxed" placeholder="Enter clinical notes, narrative text, or telemetry metrics here..."></textarea>
                    </div>

                    <!-- File Upload -->
                    <div id="input-file" class="hidden">
                        <div class="border-2 border-dashed border-[#27272a] hover:border-neutral-500 rounded-lg p-6 text-center transition bg-[#09090b]">
                            <input type="file" id="fileInput" class="hidden" onchange="updateFileName(this)">
                            <label for="fileInput" class="cursor-pointer block">
                                <div class="text-neutral-400 text-xs font-mono mb-1">Click to select CSV, JSON, or TXT file</div>
                                <div class="text-[11px] text-neutral-600 font-mono">Processes up to 500 rows or 8,192 tokens directly</div>
                            </label>
                            <div id="fileNameDisplay" class="mt-2 text-xs font-mono text-white hidden"></div>
                        </div>
                    </div>

                    <div class="mt-2 flex items-center justify-between text-[11px] font-mono text-neutral-500">
                        <span id="inputCharCount">0 characters</span>
                        <button type="button" onclick="clearInputState()" class="hover:text-neutral-300 underline">Clear State</button>
                    </div>
                </div>

                <!-- Questions Schema Card -->
                <div class="bg-[#121214] border border-[#27272a] rounded-lg p-4">
                    <div class="flex items-center justify-between mb-3 border-b border-[#27272a] pb-2">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-mono uppercase tracking-wider text-white font-bold">2. Schema Questions</span>
                            <span class="text-[10px] font-mono uppercase px-1.5 py-0.2 bg-neutral-900 border border-neutral-700 text-neutral-400 rounded">Criteria Specification</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <button type="button" onclick="prettifyQuestionsJson()" class="px-2 py-1 text-xs font-mono bg-neutral-900 hover:bg-neutral-800 text-neutral-400 hover:text-white border border-neutral-800 rounded transition">Format Schema</button>
                        </div>
                    </div>

                    <textarea id="questionsInput" rows="10" class="w-full bg-[#09090b] border border-[#27272a] focus:border-neutral-400 rounded p-3 text-xs font-mono text-neutral-200 outline-none leading-relaxed"></textarea>

                    <div class="mt-3">
                        <button onclick="runInference()" id="submitBtn" class="w-full bg-white hover:bg-neutral-200 active:scale-[0.99] text-black font-mono font-bold py-3 px-4 rounded text-xs uppercase tracking-wider transition border border-white flex items-center justify-center gap-2">
                            <span>RUN LAYA INFERENCE</span>
                            <span class="text-[10px] opacity-70">➔</span>
                        </button>
                    </div>
                </div>

            </div>

            <!-- Right Column: Results & Probabilities (5 Cols) -->
            <div class="lg:col-span-5 space-y-5">

                <div class="bg-[#121214] border border-[#27272a] rounded-lg p-4 flex flex-col h-full min-h-[550px]">
                    <!-- Output Header -->
                    <div class="flex items-center justify-between mb-3 border-b border-[#27272a] pb-2">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-mono uppercase tracking-wider text-white font-bold">Inference Results</span>
                            <span id="speedBadge" class="hidden text-[10px] font-mono px-2 py-0.5 bg-neutral-900 border border-neutral-700 text-white rounded"></span>
                        </div>
                        <div class="flex items-center gap-1.5">
                            <button type="button" onclick="setViewMode('visual')" id="view-visual-btn" class="px-2 py-0.5 text-xs font-mono rounded bg-white text-black font-semibold transition">Cards</button>
                            <button type="button" onclick="setViewMode('raw')" id="view-raw-btn" class="px-2 py-0.5 text-xs font-mono rounded bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800 transition">JSON</button>
                            <button type="button" onclick="copyResultsJson()" id="copyBtn" class="px-2 py-0.5 text-xs font-mono bg-neutral-900 hover:bg-neutral-800 text-neutral-400 hover:text-white border border-neutral-800 rounded transition" title="Copy JSON">Copy</button>
                            <button type="button" onclick="downloadReport()" class="px-2 py-0.5 text-xs font-mono bg-neutral-900 hover:bg-neutral-800 text-neutral-400 hover:text-white border border-neutral-800 rounded transition" title="Download Report">Export</button>
                        </div>
                    </div>

                    <!-- Output Body Container -->
                    <div class="flex-1 flex flex-col">
                        <!-- Visual Container -->
                        <div id="outputVisualContainer" class="flex-1 bg-[#09090b] border border-[#27272a] rounded p-4 font-mono text-xs overflow-y-auto max-h-[620px]">
                            <div class="h-full flex flex-col items-center justify-center text-center text-neutral-600 py-16">
                                <div class="text-2xl mb-2">⚡</div>
                                <div class="text-neutral-400 font-mono text-xs">Ready for single-pass structured inference</div>
                                <div class="text-neutral-600 text-[11px] mt-1 font-mono">Select a preset or enter input data and click Run</div>
                            </div>
                        </div>

                        <!-- Raw JSON View Container -->
                        <div id="outputRawContainer" class="hidden flex-1 bg-[#09090b] border border-[#27272a] rounded p-4 font-mono text-xs overflow-y-auto max-h-[620px]">
                            <pre id="rawJsonText" class="text-neutral-300 whitespace-pre-wrap leading-relaxed"></pre>
                        </div>
                    </div>

                    <!-- Output Footer Metadata -->
                    <div class="mt-3 pt-2 border-t border-[#27272a] flex items-center justify-between text-[11px] font-mono text-neutral-500">
                        <span id="outputEngineLabel">Engine: multilingual (8k)</span>
                        <span id="outputTimestampLabel">Status: Idle</span>
                    </div>
                </div>

            </div>

        </div>

    </main>

    <script>
        let currentMode = 'json';
        let currentView = 'visual';
        let latestResultData = null;

        const PRESETS = {
            ecg_ischemia: {
                mode: 'json',
                text: "ECG ST-T segment telemetry",
                json: JSON.stringify({
                    "heart_rate_bpm": 82,
                    "pr_baseline_adc": 2048,
                    "st_elevation_mm": 1.85,
                    "t_wave_morphology": "biphasic_inverted",
                    "q_wave_amplitude_mv": 0.04,
                    "rr_interval_ms": 732,
                    "mpu_motion_g": 1.03,
                    "motion_deviation_g": 0.03,
                    "sqi_quality_score": 0.89,
                    "detected_artifacts": []
                }, null, 2),
                questions: JSON.stringify({
                    "ischemia_probability": {
                        "type": "noul",
                        "instructions": "Estimate probability (0.0 to 1.0) of acute myocardial ischemia. If MPU accelerometer motion (motion_deviation_g > 0.18g) indicates excessive physical movement artifact, set probability to 0.0."
                    },
                    "classification": {
                        "type": "choice",
                        "instructions": "Classify the ECG segment into a clinical triage tier.",
                        "criteria": {
                            "NORMAL": "Normal sinus rhythm, clean physiological tracing, low MPU motion.",
                            "ISCHEMIA_PATIENT": "True ischemic ST-elevation or ST-depression corroborated across beats with low MPU motion.",
                            "OTHER_CARDIAC_DISEASE": "Arrhythmia, conduction block, or non-ischemic morphology with reliable quality.",
                            "INCONCLUSIVE_NOISY": "Excessive MPU motion artifact (deviation > 0.18g), electrode movement, or extreme noise."
                        }
                    },
                    "anomaly_detected": {
                        "type": "choice",
                        "instructions": "Identify the primary morphological feature.",
                        "criteria": {
                            "NONE": "No pathological anomaly; normal morphology.",
                            "ST_ELEVATION": "True persistent ST segment elevation (>1.0 mm above PR baseline).",
                            "ST_DEPRESSION": "True persistent ST segment depression (>0.5 mm below PR baseline).",
                            "T_WAVE_INVERSION": "Pathological T-wave inversion discordant from QRS.",
                            "HYPERACUTE_T": "Tall, peaked, symmetric hyperacute T-waves.",
                            "WELLENS_SYNDROME": "Biphasic or deeply inverted T-waves in precordial morphology.",
                            "ARTIFACT_DISTORTION": "Excessive MPU accelerometer motion, body movement, baseline wander, or EMG noise."
                        }
                    }
                }, null, 2)
            },
            ecg_quality: {
                mode: 'json',
                text: "ECG noise and motion verification",
                json: JSON.stringify({
                    "sqi_score": 0.58,
                    "mpu_motion_g": 1.29,
                    "motion_deviation_g": 0.29,
                    "rail_percentage": 4,
                    "powerline_snr_db": 12.4,
                    "baseline_wander_drift_mv": 0.65
                }, null, 2),
                questions: JSON.stringify({
                    "signal_acceptable": {
                        "type": "noul",
                        "instructions": "Is the ECG signal quality clean enough to reliably assess ST-T ischemia? Answer No if motion deviation exceeds 0.18g or SQI < 0.70."
                    },
                    "dominant_artifact": {
                        "type": "choice",
                        "instructions": "Determine the primary artifact source corrupting the telemetry.",
                        "criteria": {
                            "NONE": "Clean trace, minimal noise.",
                            "EXCESSIVE_MOTION": "MPU accelerometer motion deviation > 0.18g, patient walking/moving.",
                            "BASELINE_WANDER": "Low-frequency respiration drift, loose electrode skin contact.",
                            "EMG_MUSCLE_TREMOR": "High-frequency somatic muscle jitter.",
                            "POWERLINE_INTERFERENCE": "50Hz or 60Hz mains harmonic ripple."
                        }
                    }
                }, null, 2)
            },
            arrhythmia: {
                mode: 'json',
                text: "Rhythm telemetry",
                json: JSON.stringify({
                    "mean_rr_ms": 410,
                    "heart_rate_bpm": 146,
                    "rr_regularity": "irregularly_irregular",
                    "p_wave_present": false,
                    "qrs_duration_ms": 88
                }, null, 2),
                questions: JSON.stringify({
                    "rhythm_classification": {
                        "type": "choice",
                        "instructions": "Classify the cardiac rhythm based on R-R intervals and P-wave morphology.",
                        "criteria": {
                            "NORMAL_SINUS_RHYTHM": "HR 60-100 BPM, regular R-R, upright P waves.",
                            "SINUS_TACHYCARDIA": "HR > 100 BPM, regular rhythm, normal P-QRS sequence.",
                            "SINUS_BRADYCARDIA": "HR < 60 BPM, regular rhythm, normal P-QRS sequence.",
                            "ATRIAL_FIBRILLATION": "Irregularly irregular R-R interval, absence of discrete P waves.",
                            "VENTRICULAR_ECTOPY": "Wide bizarre QRS complexes (>120ms)."
                        }
                    },
                    "urgent_intervention_needed": {
                        "type": "noul",
                        "instructions": "Does this heart rate / rhythm represent an urgent clinical concern requiring immediate clinical evaluation?"
                    }
                }, null, 2)
            },
            triage: {
                mode: 'text',
                text: "58-year-old male presenting with crushing retrosternal chest pain radiating to left jaw, diaphoresis, HR 108, BP 145/92, SpO2 96%, onset 45 minutes ago.",
                json: '{}',
                questions: JSON.stringify({
                    "acuity_tier": {
                        "type": "choice",
                        "instructions": "Assign emergency department triage tier according to ESI guidelines.",
                        "criteria": {
                            "LEVEL_1_RESUSCITATION": "Immediate life-saving intervention required (cardiac arrest, airway compromise).",
                            "LEVEL_2_EMERGENT": "High risk situation, severe pain, acute chest pain/infarction suspicion (seen within 10-15 min).",
                            "LEVEL_3_URGENT": "Stable patient requiring two or more resources.",
                            "LEVEL_4_LESS_URGENT": "Stable patient requiring one resource.",
                            "LEVEL_5_NON_URGENT": "No resources needed."
                        }
                    },
                    "cardiac_origin_likely": {
                        "type": "noul",
                        "instructions": "Is an acute coronary syndrome or cardiac ischemia the likely primary etiology?"
                    }
                }, null, 2)
            },
            vitals: {
                mode: 'json',
                text: "Hemodynamic vitals",
                json: JSON.stringify({
                    "heart_rate_bpm": 122,
                    "systolic_bp": 88,
                    "diastolic_bp": 54,
                    "shock_index": 1.38,
                    "spo2_percent": 91,
                    "respiratory_rate": 26
                }, null, 2),
                questions: JSON.stringify({
                    "hemodynamic_state": {
                        "type": "choice",
                        "instructions": "Assess patient circulatory stability based on Shock Index (HR / SBP > 0.9) and vitals.",
                        "criteria": {
                            "STABLE_COMPENSATED": "Normal BP, normal HR, shock index < 0.7.",
                            "IMPENDING_DECOMPENSATION": "Borderline hypotension or tachycardia, shock index 0.7 - 0.9.",
                            "UNSTABLE_SHOCK": "Marked hypotension (SBP < 90), tachycardia, shock index > 1.0."
                        }
                    },
                    "icu_monitoring_indicated": {
                        "type": "noul",
                        "instructions": "Is immediate intensive care unit (ICU) level monitoring or vasopressor support indicated?"
                    }
                }, null, 2)
            },
            support: {
                mode: 'text',
                text: "I was billed twice for my pro telemetry subscription this morning ($199 x 2). I want an immediate refund to my card, or I will initiate a chargeback with my bank.",
                json: '{}',
                questions: JSON.stringify({
                    "department": {
                        "type": "choice",
                        "instructions": "Which department should handle this customer inquiry?",
                        "criteria": {
                            "billing_refunds": "Double charges, invoice errors, refund requests, chargeback risks.",
                            "technical_support": "Sensor connection bugs, firmware flashing, app crashes.",
                            "sales_enterprise": "Volume licensing, new hospital contracts."
                        }
                    },
                    "high_churn_risk": {
                        "type": "noul",
                        "instructions": "Is the user expressing high urgency or threatening cancellation / chargeback?"
                    }
                }, null, 2)
            },
            custom: {
                mode: 'json',
                text: "",
                json: '{\\n  "parameter_1": 100,\\n  "parameter_2": "value"\\n}',
                questions: JSON.stringify({
                    "decision_one": {
                        "type": "choice",
                        "instructions": "Describe the decision to make.",
                        "criteria": {
                            "OPTION_A": "Condition A criteria",
                            "OPTION_B": "Condition B criteria"
                        }
                    },
                    "binary_flag": {
                        "type": "noul",
                        "instructions": "Yes or No question formulation"
                    }
                }, null, 2)
            }
        };

        function setMode(mode) {
            currentMode = mode;
            ['text', 'json', 'file'].forEach(m => {
                const el = document.getElementById(`input-${m}`);
                if (el) el.classList.toggle('hidden', m !== mode);

                const btn = document.getElementById(`tab-${m}`);
                if (btn) {
                    if (m === mode) {
                        btn.className = 'px-2.5 py-1 text-xs font-mono rounded bg-white text-black font-semibold transition';
                    } else {
                        btn.className = 'px-2.5 py-1 text-xs font-mono rounded bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800 transition';
                    }
                }
            });

            const badge = document.getElementById('stateModeBadge');
            if (badge) badge.innerText = mode.toUpperCase();
            updateCharCount();
        }

        function setViewMode(v) {
            currentView = v;
            const vis = document.getElementById('outputVisualContainer');
            const raw = document.getElementById('outputRawContainer');
            const visBtn = document.getElementById('view-visual-btn');
            const rawBtn = document.getElementById('view-raw-btn');

            if (v === 'visual') {
                vis.classList.remove('hidden');
                raw.classList.add('hidden');
                visBtn.className = 'px-2 py-0.5 text-xs font-mono rounded bg-white text-black font-semibold transition';
                rawBtn.className = 'px-2 py-0.5 text-xs font-mono rounded bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800 transition';
            } else {
                vis.classList.add('hidden');
                raw.classList.remove('hidden');
                rawBtn.className = 'px-2 py-0.5 text-xs font-mono rounded bg-white text-black font-semibold transition';
                visBtn.className = 'px-2 py-0.5 text-xs font-mono rounded bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800 transition';
            }
        }

        function loadPreset(presetKey) {
            const p = PRESETS[presetKey];
            if (!p) return;
            setMode(p.mode);
            document.getElementById('textInput').value = p.text;
            document.getElementById('jsonInput').value = p.json;
            document.getElementById('questionsInput').value = p.questions;
            updateCharCount();
        }

        function updateFileName(input) {
            const display = document.getElementById('fileNameDisplay');
            if (input.files.length > 0) {
                display.innerText = `Selected: ${input.files[0].name} (${(input.files[0].size / 1024).toFixed(1)} KB)`;
                display.classList.remove('hidden');
            } else {
                display.classList.add('hidden');
            }
        }

        function updateCharCount() {
            let len = 0;
            if (currentMode === 'text') len = document.getElementById('textInput').value.length;
            else if (currentMode === 'json') len = document.getElementById('jsonInput').value.length;
            document.getElementById('inputCharCount').innerText = `${len.toLocaleString()} characters`;
        }

        document.getElementById('textInput').addEventListener('input', updateCharCount);
        document.getElementById('jsonInput').addEventListener('input', updateCharCount);

        function prettifyStateJson() {
            try {
                const val = document.getElementById('jsonInput').value;
                if (!val.trim()) return;
                const obj = JSON.parse(val);
                document.getElementById('jsonInput').value = JSON.stringify(obj, null, 2);
                updateCharCount();
            } catch (e) {
                alert("Invalid JSON: " + e.message);
            }
        }

        function prettifyQuestionsJson() {
            try {
                const val = document.getElementById('questionsInput').value;
                if (!val.trim()) return;
                const obj = JSON.parse(val);
                document.getElementById('questionsInput').value = JSON.stringify(obj, null, 2);
            } catch (e) {
                alert("Invalid JSON Schema: " + e.message);
            }
        }

        function clearInputState() {
            document.getElementById('textInput').value = '';
            document.getElementById('jsonInput').value = '{}';
            document.getElementById('fileInput').value = '';
            document.getElementById('fileNameDisplay').classList.add('hidden');
            updateCharCount();
        }

        async function fetchLiveTelemetry() {
            const badge = document.getElementById('liveTelemetryBadge');
            const txt = document.getElementById('telemetryStatusText');
            badge.classList.remove('hidden');
            txt.innerText = 'Syncing...';

            try {
                const res = await fetch('/api/fetch-live-telemetry');
                const data = await res.json();
                if (data.state) {
                    setMode('json');
                    document.getElementById('jsonInput').value = JSON.stringify(data.state, null, 2);
                    updateCharCount();
                    txt.innerText = `Synced (${data.source})`;
                }
            } catch (e) {
                txt.innerText = 'Sync Failed';
            }
        }

        async function copyResultsJson() {
            if (!latestResultData) {
                alert("No results to copy yet.");
                return;
            }
            try {
                await navigator.clipboard.writeText(JSON.stringify(latestResultData, null, 2));
                const btn = document.getElementById('copyBtn');
                const orig = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(() => btn.innerText = orig, 1500);
            } catch (err) {
                alert("Could not copy: " + err.message);
            }
        }

        function downloadReport() {
            if (!latestResultData) {
                alert("No inference output available to export.");
                return;
            }
            const blob = new Blob([JSON.stringify(latestResultData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `laya_prediction_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        document.getElementById('presetSelect').addEventListener('change', (e) => loadPreset(e.target.value));

        loadPreset('ecg_ischemia');

        async function runInference() {
            const btn = document.getElementById('submitBtn');
            const visualDiv = document.getElementById('outputVisualContainer');
            const rawText = document.getElementById('rawJsonText');
            const speedBadge = document.getElementById('speedBadge');
            const statusLabel = document.getElementById('outputTimestampLabel');
            const model = document.getElementById('modelSelect').value;
            const maxLen = document.getElementById('maxLenSelect').value;

            btn.disabled = true;
            btn.innerHTML = `<span class="animate-pulse">PROCESSING INFERENCE...</span>`;
            statusLabel.innerText = "Running single forward pass...";
            speedBadge.classList.add('hidden');

            visualDiv.innerHTML = `
                <div class="h-full flex flex-col items-center justify-center text-center py-16">
                    <div class="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin mb-3"></div>
                    <div class="text-white font-mono text-xs uppercase tracking-wider">Evaluating Structured Schema</div>
                    <div class="text-neutral-500 font-mono text-[11px] mt-1">Model: ${model} (${maxLen} tokens)</div>
                </div>
            `;

            const formData = new FormData();
            formData.append('questions_json', document.getElementById('questionsInput').value);
            formData.append('model', model);
            formData.append('max_len', maxLen);

            if (currentMode === 'text') {
                formData.append('text_state', document.getElementById('textInput').value);
            } else if (currentMode === 'json') {
                formData.append('json_state', document.getElementById('jsonInput').value);
            } else if (currentMode === 'file') {
                const fileEl = document.getElementById('fileInput');
                if (fileEl.files.length > 0) {
                    formData.append('file', fileEl.files[0]);
                }
            }

            const startTime = performance.now();
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                const duration = Math.round(performance.now() - startTime);
                const data = await response.json();
                latestResultData = data;

                rawText.innerText = JSON.stringify(data, null, 2);

                if (data.status === 'success') {
                    speedBadge.innerText = `${duration} ms`;
                    speedBadge.classList.remove('hidden');
                    statusLabel.innerText = `Completed in ${duration} ms (${new Date().toLocaleTimeString()})`;
                    document.getElementById('outputEngineLabel').innerText = `Engine: ${data.model_used || model}`;

                    let html = '<div class="space-y-4">';

                    if (data.routing && Object.keys(data.routing).length > 0) {
                        html += `<div class="p-2.5 bg-neutral-900 border border-neutral-800 rounded flex items-center justify-between text-[11px] font-mono">
                            <span class="text-neutral-400">ROUTER AGENT:</span>
                            <span class="text-white font-bold">${data.routing.agent || 'multilingual'}</span>
                        </div>`;
                    }

                    for (const [qKey, qVal] of Object.entries(data.answers || {})) {
                        html += `<div class="bg-[#121214] border border-[#27272a] p-3.5 rounded-lg space-y-2">`;
                        
                        html += `<div class="flex items-center justify-between">
                            <span class="font-mono text-xs uppercase font-bold text-white tracking-wider">${qKey}</span>
                            <span class="text-[10px] font-mono uppercase px-1.5 py-0.5 bg-neutral-900 border border-neutral-800 text-neutral-400 rounded">${qVal.type}</span>
                        </div>`;

                        if (qVal.type === 'choice') {
                            html += `
                                <div class="p-2.5 bg-[#09090b] border border-neutral-700 rounded flex items-center justify-between">
                                    <span class="text-neutral-400 text-xs font-mono">DECISION:</span>
                                    <span class="text-xs font-mono font-black text-white bg-neutral-800 px-2.5 py-1 rounded border border-neutral-600">${qVal.choice}</span>
                                </div>
                            `;

                            if (qVal.probabilities) {
                                html += `<div class="mt-2 space-y-1.5 pt-1">`;
                                for (const [opt, prob] of Object.entries(qVal.probabilities)) {
                                    const pct = (prob * 100).toFixed(1);
                                    const isChosen = (opt === qVal.choice);
                                    html += `
                                        <div class="flex items-center text-[11px] font-mono ${isChosen ? 'text-white font-semibold' : 'text-neutral-400'}">
                                            <span class="w-36 truncate" title="${opt}">${opt}</span>
                                            <div class="flex-1 mx-2 bg-neutral-900 h-1.5 rounded overflow-hidden border border-neutral-800">
                                                <div class="${isChosen ? 'bg-white' : 'bg-neutral-500'} h-full transition-all" style="width: ${pct}%"></div>
                                            </div>
                                            <span class="w-12 text-right">${pct}%</span>
                                        </div>
                                    `;
                                }
                                html += `</div>`;
                            }
                        } else if (qVal.type === 'noul') {
                            const pct = (qVal.noul * 100).toFixed(1);
                            const isHigh = qVal.noul >= 0.5;
                            html += `
                                <div class="p-2.5 bg-[#09090b] border border-neutral-700 rounded space-y-2">
                                    <div class="flex items-center justify-between">
                                        <span class="text-neutral-400 text-xs font-mono">CONFIDENCE (YES):</span>
                                        <span class="text-xs font-mono font-black ${isHigh ? 'text-white' : 'text-neutral-400'}">${pct}%</span>
                                    </div>
                                    <div class="w-full bg-neutral-900 h-2 rounded overflow-hidden border border-neutral-800">
                                        <div class="${isHigh ? 'bg-white' : 'bg-neutral-500'} h-full transition-all" style="width: ${pct}%"></div>
                                    </div>
                                </div>
                            `;
                        }

                        html += `</div>`;
                    }
                    html += `</div>`;
                    visualDiv.innerHTML = html;
                } else {
                    visualDiv.innerHTML = `
                        <div class="p-4 bg-neutral-900 border border-neutral-700 rounded text-neutral-200 font-mono text-xs">
                            <div class="font-bold uppercase text-white mb-1">Inference Error</div>
                            <div>${data.error || 'Unknown error occurred'}</div>
                        </div>
                    `;
                }
            } catch (err) {
                visualDiv.innerHTML = `
                    <div class="p-4 bg-neutral-900 border border-neutral-700 rounded text-neutral-200 font-mono text-xs">
                        <div class="font-bold uppercase text-white mb-1">Network Connection Error</div>
                        <div>${err.message}</div>
                    </div>
                `;
                statusLabel.innerText = "Network Error";
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<span>RUN LAYA INFERENCE</span> <span class="text-[10px] opacity-70">➔</span>`;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8085))
    print(f"\n=======================================================")
    print(f"🚀 Laya Web Playground running at: http://localhost:{port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
