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

app = FastAPI(title="Laya Interactive Playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading 8,192-token Laya Multilingual model...")
# Use Router to easily access the multilingual model with expanded 8192 token limit
router = laya.Router(preload=["multilingual"])
print("Laya model loaded!")

@app.post("/api/predict")
async def predict_endpoint(
    text_state: Optional[str] = Form(None),
    json_state: Optional[str] = Form(None),
    questions_json: str = Form(...),
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

        # Force the multilingual model and set max_len=8192
        def run_prediction():
            return router.predict(state, questions, model="multilingual", max_len=8192)

        result = await asyncio.to_thread(run_prediction)
        
        return JSONResponse({
            "status": "success",
            "state_used": state,
            "answers": result.get("answers", {}),
            "routing": result.get("routing", {})
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laya AI Interactive Playground</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    </style>
</head>
<body class="p-6 max-w-5xl mx-auto">
    <header class="mb-8 border-b border-slate-700 pb-4">
        <h1 class="text-3xl font-bold text-sky-400">⚡ Laya Model Playground</h1>
        <p class="text-slate-400 text-sm mt-1">Using: <b>laya-multilingual (8,192-token context)</b>. Upload long text, CSV data and run schema-based prompts.</p>
    </header>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Input Form -->
        <div class="space-y-5 bg-slate-800/60 p-5 rounded-xl border border-slate-700">
            <div>
                <label class="block text-sm font-semibold mb-1 text-slate-300">Preset Templates:</label>
                <select id="presetSelect" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-sky-300">
                    <option value="triage">Medical Triage (Text Input)</option>
                    <option value="ecg">ECG Telemetry (JSON / Numbers)</option>
                    <option value="support">Customer Support Ticket (Text Input)</option>
                </select>
            </div>

            <!-- Input Mode Tabs -->
            <div>
                <label class="block text-sm font-semibold mb-2 text-slate-300">Input Data (State):</label>
                <div class="flex gap-2 mb-3">
                    <button type="button" onclick="setMode('text')" id="tab-text" class="px-3 py-1 text-xs rounded bg-sky-600 text-white font-medium">Text</button>
                    <button type="button" onclick="setMode('json')" id="tab-json" class="px-3 py-1 text-xs rounded bg-slate-700 text-slate-300">JSON / Numbers</button>
                    <button type="button" onclick="setMode('file')" id="tab-file" class="px-3 py-1 text-xs rounded bg-slate-700 text-slate-300">Upload CSV / File</button>
                </div>

                <div id="input-text">
                    <textarea id="textInput" rows="5" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm font-mono text-slate-200" placeholder="Type text or clinical notes here..."></textarea>
                </div>
                <div id="input-json" class="hidden">
                    <textarea id="jsonInput" rows="5" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm font-mono text-slate-200" placeholder='{"heart_rate": 115, "st_elevation": 2.1}'></textarea>
                </div>
                <div id="input-file" class="hidden">
                    <input type="file" id="fileInput" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-300">
                    <p class="text-xs text-slate-500 mt-1">Supports large CSV/JSON/TXT files up to 8,192 tokens.</p>
                </div>
            </div>

            <!-- Questions / Criteria -->
            <div>
                <label class="block text-sm font-semibold mb-1 text-slate-300">Questions (Schema Prompts JSON):</label>
                <textarea id="questionsInput" rows="8" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm font-mono text-slate-200"></textarea>
            </div>

            <button onclick="runInference()" id="submitBtn" class="w-full bg-sky-500 hover:bg-sky-600 text-slate-950 font-bold py-3 rounded-lg transition-all shadow-lg shadow-sky-500/20">
                🚀 Run Laya Prediction
            </button>
        </div>

        <!-- Result View -->
        <div class="bg-slate-800/60 p-5 rounded-xl border border-slate-700 flex flex-col">
            <h2 class="text-lg font-bold text-slate-200 mb-3 flex justify-between items-center">
                <span>Model Output & Probabilities</span>
                <span id="speedBadge" class="text-xs font-mono bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded hidden">Ready</span>
            </h2>
            <div id="outputContainer" class="flex-1 bg-slate-900 border border-slate-700 rounded-lg p-4 font-mono text-sm overflow-y-auto text-slate-300">
                <span class="text-slate-500 italic">Click "Run Laya Prediction" to view the response...</span>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'text';

        const PRESETS = {
            triage: {
                mode: 'text',
                text: "Patient presenting with acute chest pressure radiating to left arm, HR 105, mild dyspnea.",
                json: '{"hr": 105, "chest_pain": true}',
                questions: JSON.stringify({
                    "triage_category": {
                        "type": "choice",
                        "instructions": "Determine the medical triage category.",
                        "criteria": {
                            "cardiac_emergency": "chest pain, pressure, left arm radiation, shortness of breath",
                            "respiratory": "wheezing, cough, isolated breathing difficulty",
                            "routine": "non-urgent symptoms, minor complaints"
                        }
                    },
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is immediate emergency intervention needed?"
                    }
                }, null, 2)
            },
            ecg: {
                mode: 'json',
                text: "ECG sample metrics",
                json: JSON.stringify({
                    "heart_rate_bpm": 115,
                    "st_elevation_mm": 2.1,
                    "t_wave": "inverted",
                    "rr_regularity": "regular"
                }, null, 2),
                questions: JSON.stringify({
                    "ischemia_risk": {
                        "type": "choice",
                        "instructions": "Classify the ischemia risk level based on telemetry.",
                        "criteria": {
                            "NORMAL": "st_elevation < 1.0mm, upright T-wave",
                            "WARNING": "st_elevation between 1.0mm and 2.0mm, inverted T-wave",
                            "CRITICAL": "st_elevation > 2.0mm, marked ST elevation"
                        }
                    },
                    "st_t_anomaly": {
                        "type": "noul",
                        "instructions": "Is an ST-T wave anomaly present?"
                    }
                }, null, 2)
            },
            support: {
                mode: 'text',
                text: "Hi, I was charged twice for my subscription this month. Please issue a refund immediately or I will cancel my account.",
                json: '{}',
                questions: JSON.stringify({
                    "department": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {
                            "billing": "invoices, duplicate charges, refunds",
                            "tech_support": "bugs, login issues, app crash",
                            "sales": "upgrades, new enterprise contracts"
                        }
                    },
                    "churn_risk": {
                        "type": "noul",
                        "instructions": "Does the customer threaten to cancel or leave?"
                    }
                }, null, 2)
            }
        };

        function setMode(mode) {
            currentMode = mode;
            ['text', 'json', 'file'].forEach(m => {
                document.getElementById(`input-${m}`).classList.toggle('hidden', m !== mode);
                const btn = document.getElementById(`tab-${m}`);
                if (m === mode) {
                    btn.className = 'px-3 py-1 text-xs rounded bg-sky-600 text-white font-medium';
                } else {
                    btn.className = 'px-3 py-1 text-xs rounded bg-slate-700 text-slate-300';
                }
            });
        }

        function loadPreset(presetKey) {
            const p = PRESETS[presetKey];
            if (!p) return;
            setMode(p.mode);
            document.getElementById('textInput').value = p.text;
            document.getElementById('jsonInput').value = p.json;
            document.getElementById('questionsInput').value = p.questions;
        }

        document.getElementById('presetSelect').addEventListener('change', (e) => loadPreset(e.target.value));

        loadPreset('triage');

        async function runInference() {
            const btn = document.getElementById('submitBtn');
            const outDiv = document.getElementById('outputContainer');
            const badge = document.getElementById('speedBadge');
            
            btn.disabled = true;
            btn.innerText = '⏳ Processing (Laya Model)...';
            outDiv.innerHTML = '<span class="text-sky-400">Predicting in a single forward pass...</span>';
            badge.classList.add('hidden');

            const formData = new FormData();
            formData.append('questions_json', document.getElementById('questionsInput').value);

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

                if (data.status === 'success') {
                    badge.innerText = `${duration} ms response`;
                    badge.classList.remove('hidden');
                    
                    let html = '<div class="space-y-4">';
                    for (const [qKey, qVal] of Object.entries(data.answers)) {
                        html += `<div class="bg-slate-800/90 border border-slate-700 p-3 rounded-lg">`;
                        html += `<div class="text-xs uppercase text-slate-400 font-semibold mb-1">${qKey} (${qVal.type})</div>`;
                        
                        if (qVal.type === 'choice') {
                            html += `<div class="text-lg font-bold text-sky-300">Selected: ${qVal.choice}</div>`;
                            if (qVal.probabilities) {
                                html += `<div class="mt-2 space-y-1">`;
                                for (const [opt, prob] of Object.entries(qVal.probabilities)) {
                                    const pct = (prob * 100).toFixed(1);
                                    html += `<div class="flex items-center text-xs justify-between">
                                        <span class="w-24 text-slate-400">${opt}:</span>
                                        <div class="flex-1 mx-2 bg-slate-900 h-2 rounded overflow-hidden">
                                            <div class="bg-sky-400 h-full" style="width: ${pct}%"></div>
                                        </div>
                                        <span class="w-12 text-right font-mono">${pct}%</span>
                                    </div>`;
                                }
                                html += `</div>`;
                            }
                        } else if (qVal.type === 'noul') {
                            const pct = (qVal.noul * 100).toFixed(1);
                            html += `<div class="text-base font-bold text-emerald-400">Yes Probability: ${pct}%</div>`;
                            html += `<div class="w-full bg-slate-900 h-2 rounded overflow-hidden mt-1">
                                <div class="bg-emerald-400 h-full" style="width: ${pct}%"></div>
                            </div>`;
                        }
                        html += `</div>`;
                    }
                    html += `</div>`;
                    outDiv.innerHTML = html;
                } else {
                    outDiv.innerHTML = `<span class="text-rose-400">Error: ${data.error}</span>`;
                }
            } catch (err) {
                outDiv.innerHTML = `<span class="text-rose-400">Network Error: ${err.message}</span>`;
            } finally {
                btn.disabled = false;
                btn.innerText = '🚀 Run Laya Prediction';
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
    uvicorn.run(app, host="0.0.0.0", port=8080)
