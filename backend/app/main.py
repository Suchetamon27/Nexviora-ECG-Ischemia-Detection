import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .hardware.serial_reader import SerialECGReader
from .signal.pipeline import SignalPipeline
from .inference.window import ECGWindow
from .streaming.manager import StreamManager
from .inference.ollama import OllamaECGClient
from .inference.worker import InferenceWorker
import json

app = FastAPI(title="Live ECG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream_manager = StreamManager()
reader = SerialECGReader(port=settings.serial_port, baudrate=settings.serial_baudrate)
signal_pipeline = SignalPipeline()
ecg_window = ECGWindow(
    sampling_rate=settings.sampling_rate,
    window_seconds=settings.inference_window_seconds,
    step_seconds=settings.inference_step_seconds,
)

inference_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
ollama_client = OllamaECGClient(base_url=settings.ollama_url, model=settings.ollama_model, temperature=settings.model_temperature)

ECG_SCHEMA = {
    "type": "object",
    "properties": {
        "screening_status": {
            "type": "string",
            "enum": ["possible_ischemic_pattern", "no_obvious_ischemic_pattern", "indeterminate"],
        },
        "image_quality": {"type": "string", "enum": ["adequate", "limited"]},
        "baseline": {
            "type": "object",
            "properties": {
                "heart_rate_bpm": {"type": "number"},
                "rr_interval_ms": {"type": "number"},
                "rr_regularity": {"type": "string"},
                "isoelectric_reference": {"type": "string"},
                "qrs_baseline": {"type": "string"},
                "baseline_st_t": {"type": "string"},
            },
            "required": ["heart_rate_bpm", "rr_interval_ms", "rr_regularity", "isoelectric_reference", "qrs_baseline", "baseline_st_t"],
        },
        "ischemia_parameters": {
            "type": "object",
            "properties": {
                "st_segment_deviation": {"type": "string"},
                "t_wave_abnormality": {"type": "string"},
                "q_wave_abnormality": {"type": "string"},
                "st_t_consistency": {"type": "string"},
            },
            "required": ["st_segment_deviation", "t_wave_abnormality", "q_wave_abnormality", "st_t_consistency"],
        },
        "rr_analysis": {
            "type": "object",
            "properties": {
                "mean_rr_ms": {"type": "number"},
                "regularity": {"type": "string"},
                "visible_r_peaks": {"type": "integer"},
            },
            "required": ["mean_rr_ms", "regularity", "visible_r_peaks"],
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["screening_status", "image_quality", "baseline", "ischemia_parameters", "rr_analysis", "evidence", "limitations"],
}

SYSTEM_PROMPT = """
You are an ECG research-analysis assistant.

Analyze the supplied image as a SINGLE-LEAD ECG graph. This is a research
prototype, not a clinical diagnostic system.

ECG REFERENCE KNOWLEDGE:
- The isoelectric baseline is the reference for judging ST-segment displacement.
- Myocardial ischemia/acute coronary syndromes can be associated with ST-segment
  depression/elevation and T-wave abnormalities. These findings are not by
  themselves proof of ischemia and require clinical context.
- Pathological Q waves must be reported separately because they are more
  associated with myocardial infarction/prior infarction than ischemia alone.
- A single lead cannot establish the multi-lead distribution used in clinical
  ECG interpretation and cannot reliably localize an ischemic region.
- Do not invent measurements that are not visible.
- If grid/time calibration is absent, numerical timing must be described as approximate.
- If waveform quality is insufficient, return indeterminate rather than guessing.

WITHIN-RECORDING BASELINE:
First describe the visible baseline: R-R spacing, rhythm regularity, isoelectric
reference, QRS morphology, and baseline ST/T appearance.

FOUR RESEARCH ISCHEMIA PARAMETERS:
1. ST-segment deviation relative to the visible isoelectric reference.
2. T-wave abnormality: inversion, unusual amplitude, asymmetry, or morphology.
3. Q-wave abnormality: unusually deep/wide Q morphology when clearly visible.
4. ST-T consistency across multiple consecutive beats.

SEPARATE RHYTHM PARAMETER:
R-R interval: estimate the average time between visible R peaks and describe
regularity. Do not treat RR abnormality as an ischemia marker.

STATUS:
- possible_ischemic_pattern: one or more visible findings may be compatible with
  ischemia, while acknowledging the single-lead limitation.
- no_obvious_ischemic_pattern: no obvious ischemic morphology is visible.
- indeterminate: quality, calibration, or available lead information is too limited.

Never provide a medical diagnosis. Never provide a fake probability percentage.
"""

USER_PROMPT = """
Analyze this ECG graph using the exact framework in the system instructions.

Return ONLY the requested JSON object.

Requirements:
- Start with the recording baseline.
- Analyze all four research ischemia parameters.
- Separately analyze the R-R interval.
- Distinguish observed findings from uncertainty.
- Do not invent lead names, amplitudes, time scales, or clinical history.
- If a numerical value cannot be read from the graph, use an approximate value
  only when visually reasonable; otherwise report the limitation.
"""

async def handle_inference_result(result: dict):
    try:
        if "error" in result:
            raise Exception(result["error"])
        
        raw_content = result.get("message", {}).get("content", "{}")
        data = json.loads(raw_content)
        
        is_ischemia = data.get("screening_status") == "possible_ischemic_pattern"
        
        normalized = {
            "type": "ischemia_result",
            "timestamp": datetime.utcnow().isoformat(),
            "ischemia_detected": is_ischemia,
            "probability": 1.0 if is_ischemia else 0.0,
            "finding": data.get("screening_status", "indeterminate"),
            "model": settings.ollama_model,
            "details": data
        }
    except Exception as e:
        normalized = {
            "type": "ischemia_result",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
    await stream_manager.broadcast(normalized)

worker = InferenceWorker(
    client=ollama_client,
    window_queue=inference_queue,
    result_callback=handle_inference_result,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    format_schema=ECG_SCHEMA
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hardware_mode": settings.hardware_mode,
        "sampling_rate": settings.sampling_rate,
    }


async def acquisition_loop():
    async for packet in reader.read():
        now_us = int(datetime.utcnow().timestamp() * 1e6)
        processed = signal_pipeline.process(packet, now_us)

        # Broadcast chunk to frontend
        await stream_manager.broadcast({
            "type": "ecg",
            "timestamp": now_us,
            "samples": processed.raw_ecg,
            "filtered_samples": processed.filtered_ecg,
            "ecg_hr": packet.ecgHr,
            "hr": packet.hr,
            "spo2": packet.spo2,
            "pitch": packet.pitch,
            "roll": packet.roll,
            "motion": packet.motion,
            "lo_p": packet.loP,
            "lo_m": packet.loM,
            "signal_quality": processed.signal_quality
        })

        if processed.signal_quality != "GOOD":
            continue

        window = ecg_window.add_chunk(processed.filtered_ecg)
        
        if window is None:
            continue

        if inference_queue.full():
            try:
                inference_queue.get_nowait()
                inference_queue.task_done()
            except asyncio.QueueEmpty:
                pass

        await inference_queue.put(window)


@app.on_event("startup")
async def startup():
    asyncio.create_task(acquisition_loop())
    asyncio.create_task(worker.run())

# Serve static files for frontend
from .api.websocket import router as ws_router
import os
app.include_router(ws_router)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")