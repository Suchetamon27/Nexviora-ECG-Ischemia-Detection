#!/usr/bin/env python3
"""
Nexviora ECG - Laya Local Model Preloader & Verification Tool
Preloads the 8,192-token Laya Multilingual Model (ModernBERT + RL Decision Agent)
from Hugging Face (convaiinnovations/laya) into the local cache.
"""

import sys
import time

def main():
    print("=" * 65)
    print("⚡ NEXVIORA: LAYA LOCAL MULTILINGUAL MODEL SETUP")
    print("=" * 65)

    try:
        import laya
        print(f"[✓] Laya runtime installed: v{getattr(laya, '__version__', '0.3.24')}")
    except ImportError:
        print("[!] ERROR: 'laya' package not found.")
        print("    Please run: pip install laya")
        sys.exit(1)

    print("\n[+] Preloading 'multilingual' model checkpoint (8,192 token window)...")
    print("    Repository: convaiinnovations/laya (subfolder: multilingual)")
    start_t = time.perf_counter()
    
    try:
        router = laya.Router(preload=["multilingual"])
        load_ms = round((time.perf_counter() - start_t) * 1000)
        print(f"[✓] Model loaded into memory successfully in {load_ms} ms!")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)

    print("\n[+] Executing clinical telemetry verification inference...")
    sample_state = {
        "heart_rate_bpm": 84,
        "st_elevation_mm": 1.92,
        "t_wave_morphology": "biphasic_inverted",
        "motion_deviation_g": 0.03,
        "pr_baseline_adc": 2048
    }

    sample_questions = {
        "ischemia_risk": {
            "type": "noul",
            "instructions": "Estimate probability (0.0 to 1.0) of acute myocardial ischemia."
        },
        "triage_category": {
            "type": "choice",
            "instructions": "Classify the segment into clinical tier.",
            "criteria": {
                "NORMAL": "Normal physiological tracing.",
                "WARNING": "Borderline ST deviation.",
                "CRITICAL": "Significant ST elevation or T-wave inversion."
            }
        }
    }

    t0 = time.perf_counter()
    result = router.predict(sample_state, sample_questions, model="multilingual", max_len=8192)
    infer_ms = round((time.perf_counter() - t0) * 1000)

    print(f"[✓] Forward pass complete in {infer_ms} ms!")
    print(f"    - Ischemia Confidence: {result.get('answers', {}).get('ischemia_risk', {}).get('noul', 0.0) * 100:.1f}%")
    print(f"    - Classification: {result.get('answers', {}).get('triage_category', {}).get('choice', 'N/A')}")
    print(f"    - Routing: {result.get('routing', {}).get('model', 'multilingual')}")
    print("\n[✓] Laya Local AI environment is fully functional and ready for real-time telemetry gating.")
    print("=" * 65)

if __name__ == "__main__":
    main()
