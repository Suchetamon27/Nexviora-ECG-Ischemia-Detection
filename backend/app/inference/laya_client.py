import os
import re
import json
import asyncio
import laya
from typing import Dict, Any, Optional
from openai import OpenAI
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, CLINICAL_QUESTIONS

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)
MODEL_PRIMARY = "meta/muse-glimmer-30b"


def parse_ai_response_content(raw_text: str) -> Dict[str, Any]:
    """
    Robust parser that extracts clinical fields from JSON or structured clinical section text.
    Handles standard JSON, markdown codeblocks, and plain text formats.
    """
    if not raw_text or not raw_text.strip():
        return {
            "status": "indeterminate",
            "recording_baseline": "No response returned from model.",
            "research_ischemia_parameters": {
                "st_segment_deviation": "Indeterminate",
                "t_wave_abnormality": "Indeterminate",
                "q_wave_abnormality": "Indeterminate",
                "st_t_consistency": "Indeterminate"
            },
            "r_r_interval_analysis": "Indeterminate",
            "summary": "Model returned empty response."
        }

    clean_text = raw_text.strip()
    clean_json_str = clean_text
    if "```json" in clean_json_str:
        clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_json_str:
        clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

    # Attempt JSON parsing first
    first_brace = clean_json_str.find("{")
    last_brace = clean_json_str.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = clean_json_str[first_brace:last_brace + 1]
        try:
            return json.loads(json_candidate, strict=False)
        except Exception:
            try:
                sanitized = re.sub(r'(?<!\\)\n', r'\\n', json_candidate)
                return json.loads(sanitized, strict=False)
            except Exception:
                pass

    # Fallback: Parse structured section text (e.g. SCREENING STATUS, BASELINE, FOUR ISCHEMIA PARAMETERS)
    parsed = {
        "recording_baseline": "",
        "research_ischemia_parameters": {
            "st_segment_deviation": "Normal isoelectric",
            "t_wave_abnormality": "Normal upright",
            "q_wave_abnormality": "Non-pathological",
            "st_t_consistency": "Consistent across cycles"
        },
        "r_r_interval_analysis": "",
        "status": "no_obvious_ischemic_pattern",
        "summary": ""
    }

    # Extract SCREENING STATUS
    status_match = re.search(r"SCREENING STATUS[:\s]*([^\n\r]+)", clean_text, re.IGNORECASE)
    if status_match:
        st_val = status_match.group(1).strip().lower()
        if "possible" in st_val or "ischemi" in st_val:
            parsed["status"] = "possible_ischemic_pattern"
        elif "indeterminate" in st_val or "noise" in st_val:
            parsed["status"] = "indeterminate"
        else:
            parsed["status"] = "no_obvious_ischemic_pattern"

    # Extract BASELINE section
    baseline_match = re.search(r"BASELINE[:\s]*(.*?)(?=FOUR ISCHEMIA PARAMETERS|R-R ANALYSIS|EVIDENCE|LIMITATIONS|$)", clean_text, re.IGNORECASE | re.DOTALL)
    if baseline_match:
        parsed["recording_baseline"] = baseline_match.group(1).strip()

    # Extract FOUR ISCHEMIA PARAMETERS
    params_match = re.search(r"FOUR ISCHEMIA PARAMETERS[:\s]*(.*?)(?=R-R ANALYSIS|EVIDENCE|LIMITATIONS|$)", clean_text, re.IGNORECASE | re.DOTALL)
    if params_match:
        p_text = params_match.group(1).strip()
        st_dev = re.search(r"st_segment_deviation[:\s]*([^\n\r]+)", p_text, re.IGNORECASE)
        t_abn = re.search(r"t_wave_abnormality[:\s]*([^\n\r]+)", p_text, re.IGNORECASE)
        q_abn = re.search(r"q_wave_abnormality[:\s]*([^\n\r]+)", p_text, re.IGNORECASE)
        st_con = re.search(r"st_t_consistency[:\s]*([^\n\r]+)", p_text, re.IGNORECASE)

        if st_dev:
            parsed["research_ischemia_parameters"]["st_segment_deviation"] = st_dev.group(1).strip()
        if t_abn:
            parsed["research_ischemia_parameters"]["t_wave_abnormality"] = t_abn.group(1).strip()
        if q_abn:
            parsed["research_ischemia_parameters"]["q_wave_abnormality"] = q_abn.group(1).strip()
        if st_con:
            parsed["research_ischemia_parameters"]["st_t_consistency"] = st_con.group(1).strip()

    # Extract R-R ANALYSIS
    rr_match = re.search(r"R-R ANALYSIS[:\s]*(.*?)(?=EVIDENCE|LIMITATIONS|$)", clean_text, re.IGNORECASE | re.DOTALL)
    if rr_match:
        parsed["r_r_interval_analysis"] = rr_match.group(1).strip()

    # Extract EVIDENCE & SUMMARY
    evidence_match = re.search(r"EVIDENCE[:\s]*(.*?)(?=LIMITATIONS|$)", clean_text, re.IGNORECASE | re.DOTALL)
    summary_parts = []
    if evidence_match:
        summary_parts.append(evidence_match.group(1).strip())
    
    limitations_match = re.search(r"LIMITATIONS[:\s]*(.*)", clean_text, re.IGNORECASE | re.DOTALL)
    if limitations_match:
        summary_parts.append("Limitations: " + limitations_match.group(1).strip())

    parsed["summary"] = "\n".join(summary_parts) if summary_parts else clean_text[:400]
    return parsed


class LayaClient:
    def __init__(self, model_id: str = MODEL_PRIMARY):
        print(f"[LayaClient] Initializing local Laya model ({model_id})...")
        try:
            self.local_agent = laya.load("convaiinnovations/laya")
            print("[LayaClient] Local Laya model loaded and ready.")
        except Exception as e:
            print(f"[LayaClient] Warning loading local Laya agent: {e}")
            self.local_agent = None

        print(f"[LayaClient] Initializing NVIDIA OpenAI Client ({NVIDIA_BASE_URL}, Model: {MODEL_PRIMARY}, timeout=120.0s)...")
        # 120-second timeout ensures large reasoning completions do not time out
        self.nvidia_client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
            timeout=120.0
        )
        self.glm_model_id = MODEL_PRIMARY

    async def infer_local_laya(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Runs local Laya model schema inference in background thread with MPU motion discard rule."""
        # 1. MPU Motion Check: If moving too much (|motion_g - 1.0| > 0.18), discard chunk as INCONCLUSIVE_NOISY
        motion_dev = state.get("motion_deviation_g", abs(state.get("mpu_motion_g", 1.0) - 1.0))
        artifacts = state.get("detected_artifacts", [])
        if motion_dev > 0.18 or "EXCESSIVE_MOTION" in artifacts:
            return {
                "ischemia_probability": 0.0,
                "classification": "INCONCLUSIVE_NOISY",
                "primary_anomaly": "ARTIFACT_DISTORTION",
                "discarded": True,
                "discard_reason": f"Discarded by MPU Accelerometer: Excessive Body Motion ({motion_dev:.2f}g deviation)"
            }

        if not self.local_agent:
            return {
                "ischemia_probability": 0.0,
                "classification": "NORMAL",
                "primary_anomaly": "NONE"
            }

        def _predict():
            return self.local_agent.predict(state, CLINICAL_QUESTIONS)

        try:
            raw_result = await asyncio.to_thread(_predict)
            answers = raw_result.get("answers", {})

            prob_ans = answers.get("ischemia_probability", {})
            ischemia_prob = float(prob_ans.get("noul", 0.0))

            class_ans = answers.get("classification", {})
            classification = class_ans.get("choice", "NORMAL")

            anom_ans = answers.get("anomaly_detected", {})
            primary_anomaly = anom_ans.get("choice", "NONE")

            # Double check prediction against MPU motion
            if motion_dev > 0.18:
                classification = "INCONCLUSIVE_NOISY"
                primary_anomaly = "ARTIFACT_DISTORTION"
                ischemia_prob = 0.0

            return {
                "ischemia_probability": round(ischemia_prob, 3),
                "classification": classification,
                "primary_anomaly": primary_anomaly,
                "raw_answers": answers
            }
        except Exception as e:
            print(f"[LayaClient] Local Laya prediction error: {e}")
            return {
                "ischemia_probability": 0.0,
                "classification": "NORMAL",
                "primary_anomaly": "NONE"
            }

    async def infer_nvidia_glm(
        self,
        csv_data_3s: str,
        quality_metrics: Dict[str, Any],
        features_summary: str = ""
    ) -> Dict[str, Any]:
        """Sends calibrated electrophysiological measurements + single-column CSV to NVIDIA meta/muse-glimmer-30b."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            features_summary=features_summary or "- Standard clinical metrics extracted from calibrated front-end.",
            csv_data=csv_data_3s
        )

        model_id = MODEL_PRIMARY

        def _call_api():
            completion = self.nvidia_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                top_p=0.90,
                max_tokens=3072,
                stream=False
            )
            msg = completion.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", None) or str(msg)
            return content

        try:
            print(f"[LayaClient] Calling NVIDIA API ({model_id})...")
            raw_response = await asyncio.to_thread(_call_api)
            
            if not raw_response or len(raw_response.strip()) == 0:
                raise ValueError("Received empty response from meta/muse-glimmer-30b API")

            parsed_data = parse_ai_response_content(raw_response)
            print(f"[LayaClient] Successfully received clinical analysis from {model_id}!")

            return {
                "status": "success",
                "raw_content": raw_response,
                "parsed": parsed_data,
                "model": model_id,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt
            }
        except Exception as e:
            print(f"[LayaClient] Exception with {model_id}: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "raw_content": "",
                "parsed": {
                    "recording_baseline": "Analysis interrupted due to API communication issue.",
                    "research_ischemia_parameters": {
                        "st_segment_deviation": "Indeterminate",
                        "t_wave_abnormality": "Indeterminate",
                        "q_wave_abnormality": "Indeterminate",
                        "st_t_consistency": "Indeterminate"
                    },
                    "r_r_interval_analysis": "Indeterminate",
                    "status": "indeterminate",
                    "summary": f"NVIDIA API Error: {e}"
                },
                "model": model_id,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt
            }
