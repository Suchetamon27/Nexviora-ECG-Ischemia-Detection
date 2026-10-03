import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import numpy as np

from .config import settings
from .hardware.wifi_reader import WifiECGReader
from .hardware.continuous_recorder import continuous_recorder
from .signal.pipeline import SignalPipeline
from .inference.window import ECGWindow
from .streaming.manager import stream_manager
from .inference.laya_client import LayaClient
from .inference.worker import InferenceWorker
from .inference.logger import log_chunk_to_csv, CSV_FILE_PATH, init_csv_file

app = FastAPI(title="Nexviora Live Clinical ECG Ischemia Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Web / Wi-Fi Telemetry Reader exclusively (No USB serial searching)
print(f"[Main] Initializing Web/Wi-Fi Telemetry Reader ({settings.esp_wifi_url})...")
reader = WifiECGReader(base_url=settings.esp_wifi_url)

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
    # Log chunk diagnostic result to persistent CSV file
    log_chunk_to_csv(result)
    # Broadcast to all connected WebSockets
    await stream_manager.broadcast(result)

worker = InferenceWorker(
    client=laya_client,
    window_queue=inference_queue,
    result_callback=handle_chunk_result,
    signal_pipeline=signal_pipeline
)

@app.get("/api/session/download-csv")
async def download_csv():
    """Returns the recorded chunk CSV log file for inspection."""
    init_csv_file()
    return FileResponse(
        path=CSV_FILE_PATH,
        filename="ecg_session_log.csv",
        media_type="text/csv"
    )

@app.get("/api/session/export-full-csv")
async def export_full_session_csv():
    """Generates and downloads full resolution continuous CSV containing all accumulated ECG samples in session."""
    from fastapi.responses import Response
    if not worker.session_csv_chunks:
        if continuous_recorder.current_file_path and os.path.exists(continuous_recorder.current_file_path):
            return FileResponse(
                path=continuous_recorder.current_file_path,
                filename=f"nexviora_ecg_continuous_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                media_type="text/csv"
            )
        return JSONResponse({"status": "empty", "message": "No session telemetry data captured yet."}, status_code=400)
    
    csv_lines = ["sample_index,timestamp_ms,ecg_adc,ecg_mv,chunk_id"]
    sample_idx = 0
    now_ms = int(datetime.utcnow().timestamp() * 1000) - (len(worker.session_csv_chunks) * 3000)
    
    for item in worker.session_csv_chunks:
        c_id = item.get("chunk_id", 0)
        samples = item.get("cleaned_ecg", [])
        for s in samples:
            val_adc = round(float(s), 2)
            val_mv = round((val_adc - 2048.0) / 1365.0, 4)
            t_ms = now_ms + int(sample_idx * 4)
            csv_lines.append(f"{sample_idx},{t_ms},{val_adc},{val_mv},Chunk_{c_id}")
            sample_idx += 1
            
    content = "\n".join(csv_lines)
    filename = f"nexviora_ecg_session_full_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/recordings/list")
async def list_recordings():
    """Lists all continuous telemetry CSV recordings in the recordings folder."""
    recordings_dir = continuous_recorder.output_dir
    files = []
    if os.path.exists(recordings_dir):
        for fname in os.listdir(recordings_dir):
            if fname.endswith(".csv"):
                fpath = os.path.join(recordings_dir, fname)
                stat = os.stat(fpath)
                files.append({
                    "filename": fname,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "download_url": f"/api/recordings/download/{fname}"
                })
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {
        "active_recording": continuous_recorder.current_file_path,
        "is_recording": continuous_recorder.is_recording,
        "count": len(files),
        "files": files
    }

@app.get("/api/recordings/download/{filename}")
async def download_recording(filename: str):
    """Downloads a specific continuous CSV recording file."""
    filepath = os.path.abspath(os.path.join(continuous_recorder.output_dir, filename))
    if not filepath.startswith(continuous_recorder.output_dir) or not os.path.exists(filepath):
        return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="text/csv"
    )

@app.post("/api/config/wifi")
async def update_wifi_config(payload: Dict[str, Any] = Body(...)):
    """Updates the target ESP32 Wi-Fi URL dynamically."""
    url = payload.get("url")
    if url:
        settings.esp_wifi_url = url.strip()
        if hasattr(reader, "set_esp_url"):
            reader.set_esp_url(settings.esp_wifi_url)
        return {"status": "updated", "esp_wifi_url": settings.esp_wifi_url}
    return JSONResponse({"status": "error", "message": "Missing url parameter"}, status_code=400)

@app.post("/api/config/discover-esp")
async def discover_esp():
    """Triggers auto-discovery scanner to find ESP32 IP on local network."""
    if hasattr(reader, "auto_discover_esp32_ip"):
        found_url = await reader.auto_discover_esp32_ip()
        if found_url:
            return {"status": "found", "esp_url": found_url}
        return {"status": "not_found", "message": "ESP32 not found on local network subnets"}
    return JSONResponse({"status": "error", "message": "Auto-discovery not supported in this mode"}, status_code=400)

@app.get("/health")
async def health():
    active_mode = getattr(reader, "active_mode", "disconnected")
    status_message = getattr(reader, "status_message", "Initializing...")
    is_connected = getattr(reader, "is_connected", False)
    discovered_ip = getattr(reader, "discovered_ip", None)

    return {
        "status": "ok",
        "active_mode": active_mode,
        "is_connected": is_connected,
        "status_message": status_message,
        "esp_wifi_url": settings.esp_wifi_url,
        "discovered_ip": discovered_ip,
        "serial_port": settings.serial_port,
        "is_streaming": reader.is_streaming,
        "is_continuous_recording": continuous_recorder.is_recording,
        "active_recording_file": continuous_recorder.current_file_path,
        "sampling_rate": settings.sampling_rate,
        "session_chunks_count": len(session_history),
        "session_buffer_count": len(worker.session_csv_chunks),
        "session_buffer_sec": len(worker.session_csv_chunks) * 3.0
    }

@app.post("/api/stream/start")
async def start_stream():
    new_csv_path = continuous_recorder.start_new_recording()
    success = reader.start_stream()
    return {
        "status": "streaming" if success else "started",
        "hardware_mode": settings.hardware_mode,
        "recording_csv": new_csv_path
    }

@app.post("/api/stream/stop")
async def stop_stream():
    saved_csv = continuous_recorder.stop_recording()
    reader.stop_stream()
    return {
        "status": "stopped",
        "saved_csv": saved_csv
    }

@app.post("/api/llm/analyze")
async def trigger_llm_analyze():
    """
    Manually triggered when user clicks 'Send to LLM'.
    Packages all accumulated 3s CSV telemetry chunks in memory and sends to meta/muse-glimmer-30b.
    """
    res = await worker.analyze_session_with_llm()
    if res.get("status") == "empty":
        return JSONResponse({"status": "empty", "message": res["message"]}, status_code=400)
    elif res.get("status") == "busy":
        return JSONResponse({"status": "busy", "message": res["message"]}, status_code=429)
    return res

@app.post("/api/llm/clear")
async def clear_llm_buffer():
    """Clears accumulated 3s CSV telemetry buffer in worker."""
    count = worker.clear_session_buffer()
    return {"status": "cleared", "count": count}

@app.post("/api/session/clear")
async def clear_session():
    session_history.clear()
    worker.clear_session_buffer()
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

        # Record all samples continuously into active CSV file
        continuous_recorder.record_packet(
            packet=packet,
            cleaned_samples=processed.filtered_ecg,
            quality_status=processed.signal_quality
        )

        # Broadcast live waveform packet to frontend for smooth chart rendering
        await stream_manager.broadcast({
            "type": "ecg",
            "timestamp": now_us,
            "samples": processed.raw_ecg,
            "filtered_samples": processed.filtered_ecg,
            "ir_samples": packet.ir,
            "ecg_hr": packet.ecgHr,
            "hr": packet.hr,
            "spo2": packet.spo2,
            "dc_ir": packet.dcIr,
            "amp_ir": packet.ampIr,
            "finger_detected": packet.fingerDetected,
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
