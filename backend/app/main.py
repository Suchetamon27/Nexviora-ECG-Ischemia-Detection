import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import numpy as np

from .config import settings
from .hardware.serial_reader import SerialECGReader
from .signal.pipeline import SignalPipeline
from .inference.window import ECGWindow
from .streaming.manager import StreamManager
from .inference.laya_client import LayaClient
from .inference.worker import InferenceWorker

app = FastAPI(title="Nexviora Live Clinical ECG Ischemia Backend")

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

inference_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
laya_client = LayaClient(model_id=settings.laya_model)

# Session history in backend memory
session_history: List[Dict[str, Any]] = []

async def handle_chunk_result(result: dict):
    """Callback from InferenceWorker when a 5-second chunk is processed."""
    session_history.append(result)
    # Broadcast to all connected WebSockets
    await stream_manager.broadcast(result)

worker = InferenceWorker(
    client=laya_client,
    window_queue=inference_queue,
    result_callback=handle_chunk_result,
    signal_pipeline=signal_pipeline
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hardware_mode": settings.hardware_mode,
        "serial_connected": reader._serial is not None and reader._serial.is_open,
        "is_streaming": reader.is_streaming,
        "sampling_rate": settings.sampling_rate,
        "inference_window_s": settings.inference_window_seconds,
        "noise_rejection_threshold": settings.noise_rejection_threshold,
        "session_chunks_count": len(session_history)
    }

@app.post("/api/stream/start")
async def start_stream():
    success = reader.start_stream()
    return {
        "status": "streaming" if success else "started",
        "port": settings.serial_port,
        "mock_mode": reader.mock_mode
    }

@app.post("/api/stream/stop")
async def stop_stream():
    reader.stop_stream()
    return {"status": "stopped"}

@app.post("/api/session/clear")
async def clear_session():
    session_history.clear()
    return {"status": "cleared"}

@app.post("/api/session/analyze")
async def analyze_session(payload: Optional[Dict[str, Any]] = Body(None)):
    """
    Synthesizes all chunks captured in the session and provides a finalized
    clinical diagnostic evaluation of Ischemia vs Normal vs Other Disease.
    """
    chunks = (payload.get("chunks") if payload and "chunks" in payload else None) or session_history
    if not chunks:
        return JSONResponse({
            "status": "empty",
            "message": "No session chunks recorded yet. Stream some data first!"
        }, status_code=400)

    total_chunks = len(chunks)
    clean_chunks = [c for c in chunks if c.get("pass1_status") == "CLEAN_APPROVED"]
    rejected_chunks = [c for c in chunks if c.get("pass1_status") == "REJECTED_NOISE"]

    if not clean_chunks:
        return JSONResponse({
            "verdict": "INCONCLUSIVE_HIGH_NOISE",
            "total_chunks": total_chunks,
            "clean_chunks_count": 0,
            "rejected_chunks_count": len(rejected_chunks),
            "summary": "All recorded chunks were rejected due to excessive noise or movement (>65%). Please hold still and check skin electrode adhesion before retesting."
        })

    # Tally ischemia probabilities
    probs = [c.get("ischemia_probability", 0.0) for c in clean_chunks if c.get("ischemia_probability") is not None]
    avg_ischemia_prob = float(np.mean(probs)) if probs else 0.0
    max_ischemia_prob = float(np.max(probs)) if probs else 0.0

    # Tally classifications
    classes = [c.get("classification", "NORMAL") for c in clean_chunks]
    ischemia_votes = classes.count("ISCHEMIA_PATIENT")
    normal_votes = classes.count("NORMAL")
    other_votes = classes.count("OTHER_CARDIAC_DISEASE")

    # Tally specific anomalies
    anomalies = [c.get("primary_anomaly", "NONE") for c in clean_chunks]
    anomaly_counts = {}
    for a in anomalies:
        if a and a != "NONE":
            anomaly_counts[a] = anomaly_counts.get(a, 0) + 1

    # Average vitals across clean chunks
    hrs = [c.get("hr", 72.0) for c in clean_chunks if c.get("hr")]
    spo2s = [c.get("spo2", 98.0) for c in clean_chunks if c.get("spo2")]
    avg_hr = round(float(np.mean(hrs)), 1) if hrs else 72.0
    avg_spo2 = round(float(np.mean(spo2s)), 1) if spo2s else 98.0

    # Determine final clinical verdict
    if ischemia_votes > normal_votes and (avg_ischemia_prob >= 0.50 or max_ischemia_prob >= 0.70):
        verdict = "ISCHEMIA_POSITIVE"
        verdict_title = "Potential Myocardial Ischemia Detected"
        severity = "HIGH"
        summary = (
            f"Laya AI identified recurring ischemic signatures in {ischemia_votes} of {len(clean_chunks)} clean chunks. "
            f"Mean ischemia probability is {avg_ischemia_prob * 100:.1f}% (peak: {max_ischemia_prob * 100:.1f}%). "
            f"Observed anomalies: {', '.join(f'{k} ({v}x)' for k, v in anomaly_counts.items()) or 'ST-T alterations'}."
        )
    elif other_votes > normal_votes and other_votes > ischemia_votes:
        verdict = "OTHER_CARDIAC_ANOMALY"
        verdict_title = "Non-Ischemic Cardiac Abnormality"
        severity = "MODERATE"
        summary = (
            f"Traces exhibit non-ischemic electrical deviations in {other_votes} of {len(clean_chunks)} clean chunks "
            f"(e.g., rhythm irregularity or conduction variance), with low acute ischemia likelihood ({avg_ischemia_prob * 100:.1f}%)."
        )
    else:
        verdict = "NORMAL_SINUS_RHYTHM"
        verdict_title = "Normal Physiological Rhythm"
        severity = "NORMAL"
        summary = (
            f"Evaluated {len(clean_chunks)} clean chunks. Sinus rhythm is stable without pathological ST-elevation, "
            f"ST-depression, T-wave inversion, or Wellens' syndrome pattern. Overall ischemia risk is low ({avg_ischemia_prob * 100:.1f}%)."
        )

    return JSONResponse({
        "verdict": verdict,
        "verdict_title": verdict_title,
        "severity": severity,
        "summary": summary,
        "metrics": {
            "total_chunks": total_chunks,
            "clean_chunks_count": len(clean_chunks),
            "rejected_chunks_count": len(rejected_chunks),
            "average_ischemia_probability": round(avg_ischemia_prob, 3),
            "peak_ischemia_probability": round(max_ischemia_prob, 3),
            "average_heart_rate": avg_hr,
            "average_spo2": avg_spo2,
            "class_votes": {
                "NORMAL": normal_votes,
                "ISCHEMIA_PATIENT": ischemia_votes,
                "OTHER_CARDIAC_DISEASE": other_votes
            },
            "anomaly_incidence": anomaly_counts
        }
    })

async def acquisition_loop():
    """Reads live telemetry stream from reader, streams live updates, and feeds 5s windows."""
    print("[Main] Starting continuous live acquisition loop...")
    async for packet in reader.read():
        now_us = int(datetime.utcnow().timestamp() * 1e6)
        processed = signal_pipeline.process(packet, now_us)

        # Broadcast live waveform packet to frontend for smooth chart rendering
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
            "signal_quality": processed.signal_quality,
            "chance_good_data": processed.chance_good_data,
            "chance_noise_movement": processed.chance_noise_movement
        })

        # Accumulate into 5-second chunk window (1,250 samples)
        chunk = ecg_window.add_packet(processed.filtered_ecg, processed.raw_ecg, packet)
        
        if chunk is not None:
            if inference_queue.full():
                try:
                    inference_queue.get_nowait()
                    inference_queue.task_done()
                except asyncio.QueueEmpty:
                    pass

            await inference_queue.put(chunk)

@app.on_event("startup")
async def startup():
    asyncio.create_task(acquisition_loop())
    asyncio.create_task(worker.run())

from .api.websocket import router as ws_router
app.include_router(ws_router)

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
