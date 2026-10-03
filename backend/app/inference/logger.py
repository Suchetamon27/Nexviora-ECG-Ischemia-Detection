import os
import csv
import json
from datetime import datetime
from typing import Dict, Any

from ..config import settings

CSV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ecg_session_log.csv"))

CSV_HEADERS = [
    "chunk_id",
    "timestamp",
    "pass1_status",
    "quality_status",
    "quality_score",
    "rejection_reason",
    "detected_artifacts",
    "noise_percentage",
    "chance_good_data",
    "heart_rate_bpm",
    "spo2_pct",
    "motion_g",
    "pitch_deg",
    "roll_deg",
    "snr_db",
    "p_sqi",
    "k_sqi",
    "bas_sqi",
    "hf_sqi",
    "clip_ratio",
    "st_elevation_mm",
    "st_depression_mm",
    "t_wave_morphology",
    "hyperacute_t_warning",
    "wellens_syndrome_warning",
    "ischemia_probability",
    "classification",
    "primary_anomaly",
    "filter_config",
    "ai_system_prompt",
    "ai_input_state_json",
    "ai_raw_response_json"
]

def init_csv_file():
    """Ensures the CSV log file exists immediately and has proper column headers."""
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"[CSVLogger] Initialized CSV log file at {CSV_FILE_PATH}")

# Call init_csv_file immediately on module import
init_csv_file()

def log_chunk_to_csv(result: Dict[str, Any]):
    """Appends a 5-second chunk evaluation result to the CSV file."""
    try:
        init_csv_file()
        
        clinical = result.get("clinical_features", {}) or {}
        raw_state = result.get("raw_state", {}) or {}
        sqi = result.get("sqi_metrics", {}) or {}
        
        prompt_text = result.get("ai_prompt", "")
        input_state_json = json.dumps(result.get("ai_input_state", raw_state), default=str)
        raw_response_json = json.dumps(result.get("ai_raw_response", {}), default=str)
        
        artifacts_str = ";".join(result.get("detected_artifacts", [])) if result.get("detected_artifacts") else "NONE"
        filter_cfg_str = f"fs={settings.sampling_rate},notch={settings.notch_hz}Hz,adaptive={settings.notch_adaptive},lp={settings.lowpass_hz}Hz,baseline={settings.baseline_method}"

        row = [
            result.get("chunk_id", ""),
            result.get("timestamp", datetime.utcnow().isoformat()),
            result.get("pass1_status", ""),
            result.get("quality_status", "UNKNOWN"),
            result.get("quality_score", 0.0),
            result.get("rejection_reason", "") or "N/A",
            artifacts_str,
            result.get("noise_percentage", 0.0),
            result.get("chance_good_data", 100.0),
            result.get("hr", 72.0),
            result.get("spo2", 98.0),
            result.get("motion", 1.0),
            result.get("pitch", 0.0),
            result.get("roll", 0.0),
            sqi.get("snr_db", 0.0),
            sqi.get("p_sqi", 0.0),
            sqi.get("k_sqi", 0.0),
            sqi.get("bas_sqi", 0.0),
            sqi.get("hf_sqi", 0.0),
            sqi.get("clip_ratio", 0.0),
            clinical.get("st_elevation_mm", 0.0),
            clinical.get("st_depression_mm", 0.0),
            clinical.get("t_wave_morphology", "NORMAL"),
            clinical.get("hyperacute_t_wave", False),
            clinical.get("wellens_syndrome", False),
            result.get("ischemia_probability", "") if result.get("ischemia_probability") is not None else "N/A",
            result.get("classification", ""),
            result.get("primary_anomaly", ""),
            filter_cfg_str,
            prompt_text,
            input_state_json,
            raw_response_json
        ]
        
        with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    except Exception as exc:
        print(f"[CSVLogger] Error writing chunk to CSV: {exc}")
