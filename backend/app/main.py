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

async def handle_inference_result(result: dict):
    # Normalize to backend detection result
    normalized = {
        "type": "ischemia_result",
        "timestamp": datetime.utcnow().isoformat(),
        "ischemia_detected": result.get("ischemia_detected", False),
        "probability": result.get("probability", 0.0),
        "finding": result.get("message", {}).get("content", "Analysis complete"),
        "model": settings.ollama_model
    }
    await stream_manager.broadcast(normalized)

worker = InferenceWorker(
    client=ollama_client,
    window_queue=inference_queue,
    result_callback=handle_inference_result,
    system_prompt="Analyze this ECG for ischemia.",
    user_prompt="Return JSON with ischemia_detected and probability."
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
app.include_router(ws_router)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")