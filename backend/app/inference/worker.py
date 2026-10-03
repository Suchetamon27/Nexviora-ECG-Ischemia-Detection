import asyncio
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional

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
        self.session_csv_chunks: List[Dict[str, Any]] = []
        self.is_llm_busy: bool = False

    def clear_session_buffer(self) -> int:
        count = len(self.session_csv_chunks)
        self.session_csv_chunks.clear()
        print(f"[InferenceWorker] Cleared session buffer ({count} chunks removed).")
        return count

    async def analyze_session_with_llm(self) -> Dict[str, Any]:
        """
        Manually triggered when user clicks 'Send to LLM'.
        Combines all accumulated passing 3s CSV telemetry chunks in memory into a single
        multi-window dataset and sends it to NVIDIA meta/muse-glimmer-30b model in one pass.
        """
        if self.is_llm_busy:
            return {
                "status": "busy",
                "message": "NVIDIA meta/muse-glimmer-30b API call is currently in progress..."
            }

        if not self.session_csv_chunks:
            return {
                "status": "empty",
                "message": "No passing 3-second telemetry chunks stored in session buffer yet."
            }

        self.is_llm_busy = True
        batch = list(self.session_csv_chunks)
        n_chunks = len(batch)
        total_sec = n_chunks * 3.0
        chunk_ids_str = ", ".join([f"#{item['chunk_id']}" for item in batch])

        print(f"[InferenceWorker] 'Send to LLM' triggered! Packaging {n_chunks} chunk(s) ({total_sec:.1f}s telemetry) for meta/muse-glimmer-30b...")

        # Collect all cleaned ecg samples from the session chunks
        all_cleaned_ecg = []
        for item in batch:
            all_cleaned_ecg.extend(item.get("cleaned_ecg", []))

        last_chunk = batch[-1]
        # Extract comprehensive calibrated electrophysiological measurements
        features = extract_ecg_clinical_features(
            ecg_samples=all_cleaned_ecg if all_cleaned_ecg else [2048.0] * 750,
            hr=last_chunk.get("hr", 72.0),
            fs=250
        )

        features_summary = f"""- Measured Heart Rate: {features['heart_rate_bpm']:.1f} BPM (Mean R-R interval: {features['mean_rr_ms']:.1f} ms, Rhythm: {features['rhythm_regularity']})
- Isoelectric PR Baseline: {features['isoelectric_baseline_adc']:.1f} ADC counts ({features['isoelectric_baseline_volts']:.3f} V)
- Peak R-Wave Amplitude: {features['r_peak_amplitude_adc']:.1f} ADC counts (+{features['r_peak_amplitude_mv']:.3f} mV above baseline)
- ST-Segment Deviation (J+60ms): {features['st_deviation_adc']:+.1f} ADC counts ({features['st_deviation_mv']:+.3f} mV / {features['st_deviation_mm']:+.2f} mm) -> Status: {features['st_status']}
- T-Wave Amplitude & Polarity: {features['t_wave_amplitude_adc']:+.1f} ADC counts ({features['t_wave_amplitude_mv']:+.3f} mV) -> Morphology: {features['t_wave_morphology']}
- Pathological Q-Wave: {'PRESENT (Pathological)' if features['pathological_q_wave'] else 'ABSENT (Normal)'} (Depth: {features['q_wave_depth_mv']:.3f} mV, Duration: {features['q_wave_duration_ms']:.1f} ms)
- QRS Complex Duration: ~{features['qrs_duration_ms']} ms (Normal intraventricular conduction: 60-110 ms)
- Beat-to-Beat ST-T Consistency: {features['st_consistency_pct']:.1f}% morphological concordance across {features['total_beats_detected']} detected cycles
- Signal Quality Index (SQI): {last_chunk.get('good_pct', 85.0):.1f}% Clean (High Quality)"""

        # Build single-column float ECG sample data block (subsampled at 50Hz = 20ms per sample)
        sample_slice = all_cleaned_ecg[:250 * min(12, max(3, int(total_sec)))] if all_cleaned_ecg else [2048.0] * 150
        csv_subsampled = []
        for s_i in range(0, len(sample_slice), 5):
            csv_subsampled.append(f"{round(float(sample_slice[s_i]), 2)}")
        telemetry_prompt_csv = "\n".join(csv_subsampled)

        try:
            first_qa_metrics = batch[0]["qa"].metrics if "qa" in batch[0] else {}
            glm_res = await self.client.infer_nvidia_glm(
                csv_data_3s=telemetry_prompt_csv,
                quality_metrics=first_qa_metrics,
                features_summary=features_summary
            )

            parsed_glm = glm_res.get("parsed", {})
            raw_glm = glm_res.get("raw_content", "")

            last_chunk = batch[-1]
            status_str = (parsed_glm.get("status") or "indeterminate").upper()
            params_dict = parsed_glm.get("research_ischemia_parameters") or {}
            laya_dict = last_chunk.get("local_laya") or {}

            formatted_report_text = f"""================================================================================
             NEXVIORA CLINICAL TELEMETRY & ISCHEMIA EVALUATION REPORT           
================================================================================
Generated At          : {datetime.utcnow().isoformat()}
AI Inference Model    : meta/muse-glimmer-30b (NVIDIA Integration)
Evaluated Telemetry   : {n_chunks} Continuous 3.0s Windows ({total_sec:.1f} Seconds Total)
Hardware Front-End    : AD8232 (Gain: 1100x, ADC: ESP32 12-Bit 0-4095)
Sampling Rate         : 250 Hz Continuous (50 Hz Subsampled Telemetry)
Signal Quality (SQI)  : {last_chunk.get('good_pct', 85.0):.1f}% Clean (Pass 1 Quality Gate >= 70%)
--------------------------------------------------------------------------------

SCREENING STATUS
{status_str}

BASELINE
heart_rate_bpm: {features['heart_rate_bpm']:.1f}
rr_interval_ms: {features['mean_rr_ms']:.1f}
rr_regularity: {features['rhythm_regularity']}
isoelectric_reference: {features['isoelectric_baseline_adc']:.1f} ADC counts ({features['isoelectric_baseline_volts']:.3f} V)
qrs_amplitude_mv: +{features['r_peak_amplitude_mv']:.3f} mV
qrs_duration_ms: ~{features['qrs_duration_ms']} ms
baseline_st_t: {parsed_glm.get('recording_baseline', 'Isoelectric baseline centered at ~2048 counts')}

FOUR ISCHEMIA PARAMETERS
st_segment_deviation: {params_dict.get('st_segment_deviation', f"{features['st_deviation_mv']:+.3f} mV ({features['st_status']})")}
t_wave_abnormality: {params_dict.get('t_wave_abnormality', f"{features['t_wave_amplitude_mv']:+.3f} mV ({features['t_wave_morphology']})")}
q_wave_abnormality: {params_dict.get('q_wave_abnormality', 'PRESENT' if features['pathological_q_wave'] else 'ABSENT')}
st_t_consistency: {params_dict.get('st_t_consistency', f"{features['st_consistency_pct']:.1f}% concordance")}

R-R ANALYSIS
mean_rr_ms: {features['mean_rr_ms']:.1f}
heart_rate_bpm: {features['heart_rate_bpm']:.1f}
regularity: {features['rhythm_regularity']}
visible_r_peaks: {features['total_beats_detected']}

LOCAL LAYA VALIDATION & MPU MOTION CHECK
Category: {laya_dict.get('classification', 'NORMAL')}
Primary Anomaly: {laya_dict.get('primary_anomaly', 'NONE')}
Ischemia Probability: {(laya_dict.get('ischemia_probability', 0.0) * 100):.1f}%
MPU Accelerometer Motion: {last_chunk.get('motion', 1.0):.2f}g (Pitch: {last_chunk.get('pitch', 0.0):.1f}°, Roll: {last_chunk.get('roll', 0.0):.1f}°)
Discarded by Motion: {'YES' if laya_dict.get('discarded') else 'NO'}

EVIDENCE & CLINICAL SYNTHESIS
{parsed_glm.get('summary', raw_glm or 'Analysis completed.')}

HARDWARE & CALIBRATION CONTEXT
- Hardware Front-End: AD8232 Analog Front-End (Total Gain: 1100x, ADC: ESP32 12-bit 0-4095)
- Voltage Calibration: 1.0 mV input = 1365 ADC counts (0.1 mV / 1.0 mm = 136.5 ADC counts)
- Isoelectric Baseline Reference: PR segment baseline (~2048 counts)
- Time Calibration: 20 ms / sample (50 Hz subsampled continuous telemetry)
- Optical Pulse & SpO2: {last_chunk.get('spo2', 98.0):.1f}% SpO2
================================================================================
"""

            final_payload = {
                "type": "chunk_result",
                "chunk_id": f"SESSION_{n_chunks}CHUNKS",
                "timestamp": datetime.utcnow().isoformat(),
                "pass1_status": "CLEAN_APPROVED",
                "quality_status": "CLEAN",
                "quality_score": last_chunk.get("quality_score", 1.0),
                "rejection_reason": None,
                "detected_artifacts": last_chunk.get("detected_artifacts", []),
                "sqi_metrics": first_qa_metrics,
                "noise_percentage": last_chunk.get("noise_pct", 0.0),
                "chance_good_data": last_chunk.get("good_pct", 100.0),
                "image_b64": last_chunk.get("image_b64", ""),
                "hr": last_chunk.get("hr", 72.0),
                "spo2": last_chunk.get("spo2", 98.0),
                "motion": last_chunk.get("motion", 1.0),
                "pitch": last_chunk.get("pitch", 0.0),
                "roll": last_chunk.get("roll", 0.0),
                "local_laya": laya_dict,
                "glm_result": parsed_glm,
                "raw_glm_output": raw_glm,
                "formatted_report_text": formatted_report_text,
                "ai_prompt": glm_res.get("user_prompt", ""),
                "system_prompt": glm_res.get("system_prompt", ""),
                "ai_model": f"meta/muse-glimmer-30b ({n_chunks} Windows / {total_sec:.1f}s)",
                "waiting_for_ai": False,
                "batch_chunk_count": n_chunks,
                "total_seconds": total_sec,
                "csv_data_3s": telemetry_prompt_csv
            }

            await self.result_callback(final_payload)
            return {
                "status": "success",
                "batch_chunk_count": n_chunks,
                "total_seconds": total_sec,
                "payload": final_payload
            }
        except Exception as e:
            print(f"[InferenceWorker] Error calling meta/muse-glimmer-30b API: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self.is_llm_busy = False

    async def run(self) -> None:
        print("[InferenceWorker] Telemetry graph worker loop started.")
        while True:
            chunk = await self.window_queue.get()
            self.chunk_counter += 1
            chunk_id = self.chunk_counter

            try:
                raw_ecg_data = chunk.get("raw_ecg", [])
                ecg_data = chunk.get("ecg", [])
                motion_data = chunk.get("motion", [])
                lead_off_data = chunk.get("lead_off", [])
                timestamps_data = chunk.get("timestamps", [])
                hr = chunk.get("hr", 72.0)
                spo2 = chunk.get("spo2", 98.0)
                pitch = chunk.get("pitch", 0.0)
                roll = chunk.get("roll", 0.0)
                latest_motion = chunk.get("motion_latest", 1.0)

                # -------------------------------------------------------------
                # DEEP CLINICAL CLEANING & INSTANT GRAPH PLOTTING
                # -------------------------------------------------------------
                source_ecg = raw_ecg_data if (raw_ecg_data and len(raw_ecg_data) >= 200) else ecg_data
                cleaned_ecg = self.signal_pipeline.filter.clean_5s_chunk(source_ecg)

                title_str = f"Chunk #{chunk_id} (3.0s Cleaned) · HR: {hr:.1f} BPM"
                image_b64 = generate_ecg_strip_image(cleaned_ecg, fs=250, title=title_str)

                # Generate single-column float ECG string (subsampled to 50 Hz = 150 points for 3.0s window)
                csv_lines = []
                n_samples = len(cleaned_ecg)
                for i in range(0, n_samples, 5):
                    val = round(float(cleaned_ecg[i]), 2)
                    csv_lines.append(f"{val}")
                csv_data_3s = "\n".join(csv_lines)

                # -------------------------------------------------------------
                # PASS 1: Multi-Metric Signal Quality Assessment & 70% Gate
                # -------------------------------------------------------------
                qa = self.signal_pipeline.evaluate_5s_window_quality(
                    raw_ecg_samples=source_ecg,
                    cleaned_ecg_samples=cleaned_ecg,
                    motion_samples=motion_data,
                    lead_off_flags=lead_off_data
                )

                good_pct = round(qa.quality_score * 100.0, 1)
                noise_pct = round((1.0 - qa.quality_score) * 100.0, 1)

                motion_dev = abs(latest_motion - 1.0)
                clinical_state = extract_ecg_clinical_features(cleaned_ecg, hr=hr, fs=250)
                clinical_state["signal_quality_score"] = qa.quality_score
                clinical_state["signal_quality_status"] = qa.status
                clinical_state["detected_artifacts"] = qa.detected_artifacts
                clinical_state["snr_db"] = qa.metrics.get("snr_db", 15.0)
                clinical_state["noise_percentage"] = noise_pct
                clinical_state["motion_g"] = latest_motion
                clinical_state["mpu_motion_g"] = latest_motion
                clinical_state["motion_deviation_g"] = motion_dev
                clinical_state["mpu_pitch_deg"] = pitch
                clinical_state["mpu_roll_deg"] = roll
                clinical_state["spo2_pct"] = spo2

                local_laya_res = await self.client.infer_local_laya(clinical_state)

                # 70% CLEAN DATA GATE
                pass1_approved = (good_pct >= 70.0) and not qa.is_rejected

                if not pass1_approved:
                    rejection_msg = qa.rejection_reason or f"Signal Quality {good_pct}% is below required 70.0% clean threshold"
                    result_payload = {
                        "type": "chunk_result",
                        "chunk_id": chunk_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "pass1_status": "REJECTED_NOISE",
                        "quality_status": "POOR_QUALITY",
                        "quality_score": qa.quality_score,
                        "rejection_reason": rejection_msg,
                        "detected_artifacts": qa.detected_artifacts,
                        "noise_percentage": noise_pct,
                        "chance_good_data": good_pct,
                        "chance_noise_movement": noise_pct,
                        "image_b64": image_b64,
                        "hr": hr,
                        "spo2": spo2,
                        "motion": latest_motion,
                        "pitch": pitch,
                        "roll": roll,
                        "local_laya": local_laya_res,
                        "glm_result": None,
                        "ai_model": "meta/muse-glimmer-30b (SKIPPED: Quality < 70%)",
                        "csv_preview": csv_data_3s[:200] + "...",
                        "session_buffer_count": len(self.session_csv_chunks)
                    }
                    # INSTANTLY BROADCAST GRAPH PLOT FOR REJECTED/FAILED CHUNK TO SECTION 2
                    await self.result_callback(result_payload)
                else:
                    # APPROVED -> STORE CSV DATA IN SESSION BUFFER FOR MANUAL 'SEND TO LLM' TRIGGER
                    chunk_item = {
                        "chunk_id": chunk_id,
                        "csv_data_3s": csv_data_3s,
                        "cleaned_ecg": list(cleaned_ecg),
                        "qa": qa,
                        "quality_score": qa.quality_score,
                        "good_pct": good_pct,
                        "noise_pct": noise_pct,
                        "image_b64": image_b64,
                        "hr": hr,
                        "spo2": spo2,
                        "motion": latest_motion,
                        "pitch": pitch,
                        "roll": roll,
                        "local_laya": local_laya_res,
                        "detected_artifacts": qa.detected_artifacts
                    }
                    self.session_csv_chunks.append(chunk_item)
                    buffer_count = len(self.session_csv_chunks)
                    total_buffer_sec = buffer_count * 3.0

                    intermediate_payload = {
                        "type": "chunk_result",
                        "chunk_id": chunk_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "pass1_status": "PASSED_STORED",
                        "quality_status": "CLEAN",
                        "quality_score": qa.quality_score,
                        "rejection_reason": None,
                        "detected_artifacts": qa.detected_artifacts,
                        "noise_percentage": noise_pct,
                        "chance_good_data": good_pct,
                        "image_b64": image_b64,
                        "hr": hr,
                        "spo2": spo2,
                        "motion": latest_motion,
                        "pitch": pitch,
                        "roll": roll,
                        "local_laya": local_laya_res,
                        "glm_result": None,
                        "waiting_for_ai": True,
                        "session_buffer_count": buffer_count,
                        "session_buffer_sec": total_buffer_sec,
                        "ai_status_message": f"✓ Pass 1 SQI ({good_pct}%) & Local Laya Passed. Stored in session buffer ({buffer_count} chunks / {total_buffer_sec:.1f}s ready for 'Send to LLM')."
                    }
                    # INSTANTLY BROADCAST GRAPH PLOT TO SECTION 2
                    await self.result_callback(intermediate_payload)

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
