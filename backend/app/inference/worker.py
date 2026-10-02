import asyncio
from datetime import datetime
from typing import Callable, Dict, Any

from .laya_client import LayaClient
from ..signal.plotter import generate_ecg_strip_image
from ..signal.features import extract_ecg_clinical_features


class InferenceWorker:
    def __init__(
        self,
        client: LayaClient,
        window_queue: asyncio.Queue,
        result_callback: Callable,
        signal_pipeline,
    ) -> None:
        self.client = client
        self.window_queue = window_queue
        self.result_callback = result_callback
        self.signal_pipeline = signal_pipeline
        self.chunk_counter = 0

    async def run(self) -> None:
        print("[InferenceWorker] Worker loop started.")
        while True:
            chunk = await self.window_queue.get()
            self.chunk_counter += 1
            chunk_id = self.chunk_counter

            try:
                ecg_data = chunk.get("ecg", [])
                motion_data = chunk.get("motion", [])
                lead_off_data = chunk.get("lead_off", [])
                hr = chunk.get("hr", 72.0)
                spo2 = chunk.get("spo2", 98.0)
                pitch = chunk.get("pitch", 0.0)
                roll = chunk.get("roll", 0.0)
                latest_motion = chunk.get("motion_latest", 1.0)

                # Generate high-resolution ECG strip image for visual inspection
                title_str = f"Chunk #{chunk_id} (5.0s) · HR: {hr:.1f} BPM"
                image_b64 = generate_ecg_strip_image(ecg_data, fs=250, title=title_str)

                # -------------------------------------------------------------
                # PASS 1: Quality and Noise Gate (60-70% noise threshold window)
                # -------------------------------------------------------------
                quality_info = self.signal_pipeline.evaluate_5s_window_quality(
                    ecg_data, motion_data, lead_off_data
                )

                if quality_info["is_rejected"]:
                    # REJECTED IN PASS 1: Skip Pass 2 Laya model evaluation
                    result_payload = {
                        "type": "chunk_result",
                        "chunk_id": chunk_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "pass1_status": "REJECTED_NOISE",
                        "rejection_reason": quality_info["rejection_reason"],
                        "noise_percentage": quality_info["noise_percentage"],
                        "chance_good_data": quality_info["chance_good_data"],
                        "chance_noise_movement": quality_info["chance_noise_movement"],
                        "image_b64": image_b64,
                        "hr": hr,
                        "spo2": spo2,
                        "motion": latest_motion,
                        "pitch": pitch,
                        "roll": roll,
                        "ischemia_probability": None,
                        "classification": "REJECTED_NOISE",
                        "primary_anomaly": "HIGH_NOISE",
                        "raw_state": {
                            "rejection_reason": quality_info["rejection_reason"],
                            "noise_percentage": quality_info["noise_percentage"],
                            "chance_noise_movement": quality_info["chance_noise_movement"],
                            "motion_g": latest_motion,
                            "signal_std": quality_info["signal_std"]
                        }
                    }
                    await self.result_callback(result_payload)
                    continue

                # -------------------------------------------------------------
                # PASS 2: Clinical Ischemia Inference via Laya Model
                # -------------------------------------------------------------
                # Extract clinical electrocardiological features
                clinical_state = extract_ecg_clinical_features(ecg_data, hr=hr, fs=250)
                clinical_state["noise_percentage"] = quality_info["noise_percentage"]
                clinical_state["motion_g"] = latest_motion
                clinical_state["spo2_pct"] = spo2

                # Run Laya prediction
                laya_result = await self.client.infer(clinical_state)

                result_payload = {
                    "type": "chunk_result",
                    "chunk_id": chunk_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "pass1_status": "CLEAN_APPROVED",
                    "rejection_reason": None,
                    "noise_percentage": quality_info["noise_percentage"],
                    "chance_good_data": quality_info["chance_good_data"],
                    "chance_noise_movement": quality_info["chance_noise_movement"],
                    "image_b64": image_b64,
                    "hr": hr,
                    "spo2": spo2,
                    "motion": latest_motion,
                    "pitch": pitch,
                    "roll": roll,
                    "ischemia_probability": laya_result["ischemia_probability"],
                    "classification": laya_result["classification"],
                    "class_probabilities": laya_result["class_probabilities"],
                    "primary_anomaly": laya_result["primary_anomaly"],
                    "clinical_features": {
                        "st_elevation_mm": clinical_state["st_elevation_mm"],
                        "st_depression_mm": clinical_state["st_depression_mm"],
                        "t_wave_morphology": clinical_state["t_wave_morphology"],
                        "hyperacute_t_wave": clinical_state["hyperacute_t_wave"],
                        "wellens_syndrome": clinical_state["wellens_syndrome"],
                        "r_peak_count": clinical_state["r_peak_count"]
                    },
                    "raw_state": clinical_state
                }

                await self.result_callback(result_payload)

            except Exception as exc:
                print(f"[InferenceWorker] Error processing chunk #{chunk_id}: {exc}")
                await self.result_callback({
                    "type": "chunk_result",
                    "chunk_id": chunk_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(exc),
                    "pass1_status": "ERROR"
                })

            finally:
                self.window_queue.task_done()
