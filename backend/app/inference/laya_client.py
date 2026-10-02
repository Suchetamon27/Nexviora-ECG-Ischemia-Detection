import laya
import asyncio
from typing import Dict, Any

CLINICAL_QUESTIONS = {
    "ischemia_probability": {
        "type": "noul",
        "instructions": (
            "Clinical ECG diagnostic criteria for myocardial ischemia: "
            "1. ST elevation >= 1mm (convex or concave). "
            "2. ST depression >= 0.5mm (horizontal or downsloping). "
            "3. T-wave inversion (symmetrical, >= 1mm). "
            "4. Hyperacute T-waves (tall, broad-based, peaked). "
            "5. Wellens' syndrome (biphasic or deeply inverted precordial T-waves). "
            "Based on the provided ECG telemetry state and downsampled morphology, "
            "what is the probability (0.0 to 1.0) that this reading shows myocardial ischemia?"
        )
    },
    "classification": {
        "type": "choice",
        "instructions": "Classify the cardiac reading into the appropriate clinical category.",
        "criteria": {
            "NORMAL": "Normal sinus rhythm, physiological ST-T complexes without ischemic deviation.",
            "ISCHEMIA_PATIENT": "Positive for ischemia (marked ST elevation, ST depression, T-wave inversion, hyperacute T, or Wellens' syndrome).",
            "OTHER_CARDIAC_DISEASE": "Non-ischemic cardiac abnormality (e.g. conduction defect, arrhythmia, ectopic beats)."
        }
    },
    "anomaly_detected": {
        "type": "choice",
        "instructions": "Identify the primary specific morphological anomaly present in the ECG reading.",
        "criteria": {
            "ST_ELEVATION": "ST-segment elevation above baseline.",
            "ST_DEPRESSION": "Horizontal or downsloping ST depression.",
            "T_WAVE_INVERSION": "Symmetrical or deep inverted T-wave.",
            "HYPERACUTE_T": "Tall, peaked, broad-based hyperacute T-wave.",
            "WELLENS_SYNDROME": "Biphasic or deeply inverted precordial T-wave pattern.",
            "NONE": "No ischemic ST-T anomalies present."
        }
    }
}

class LayaClient:
    def __init__(self, model_id: str = "convaiinnovations/laya"):
        print(f"[LayaClient] Preloading local Laya agent ({model_id})...")
        self.agent = laya.load(model_id)
        print("[LayaClient] Local Laya model ready for inference!")

    async def infer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs clinical schema prediction in a background thread to prevent blocking the event loop.
        """
        def _predict():
            return self.agent.predict(state, CLINICAL_QUESTIONS)

        raw_result = await asyncio.to_thread(_predict)
        answers = raw_result.get("answers", {})

        # Extract noul probability
        prob_ans = answers.get("ischemia_probability", {})
        ischemia_prob = float(prob_ans.get("noul", 0.0))

        # Extract classification
        class_ans = answers.get("classification", {})
        classification = class_ans.get("choice", "NORMAL")
        class_probs = class_ans.get("probabilities", {})

        # Extract anomaly
        anom_ans = answers.get("anomaly_detected", {})
        primary_anomaly = anom_ans.get("choice", "NONE")

        return {
            "ischemia_probability": round(ischemia_prob, 3),
            "classification": classification,
            "class_probabilities": {k: round(v, 3) for k, v in class_probs.items()},
            "primary_anomaly": primary_anomaly,
            "raw_answers": answers
        }
