import laya

print("Loading local Laya model (convaiinnovations/laya)...")
agent = laya.load("convaiinnovations/laya")
print("Model loaded successfully!\n")

# Example 1: Text / Email Triage
print("--- Example 1: Text Triage & Risk Assessment ---")
state_text = "Patient presenting with acute chest pressure radiating to left arm, HR 105, mild dyspnea."

questions_text = {
    "triage_category": {
        "type": "choice",
        "instructions": "Determine the medical triage category.",
        "criteria": {
            "cardiac_emergency": "chest pain, pressure, left arm radiation, shortness of breath",
            "respiratory": "wheezing, cough, isolated breathing difficulty",
            "routine": "non-urgent symptoms, minor complaints"
        }
    },
    "urgent": {
        "type": "noul",
        "instructions": "Is immediate emergency intervention needed?"
    }
}

result_1 = agent.predict(state_text, questions_text)
print("State:", state_text)
print("Triage Category Choice:", result_1["answers"]["triage_category"]["choice"])
print("Probabilities:", result_1["answers"]["triage_category"]["probabilities"])
print("Urgent (Yes Probability):", result_1["answers"]["urgent"]["noul"])
print()

# Example 2: Structured Telemetry JSON Decision
print("--- Example 2: Structured ECG Metric Decision ---")
state_json = {
    "heart_rate_bpm": 115,
    "st_elevation_mm": 2.1,
    "t_wave": "inverted",
    "rr_interval_regularity": "regular"
}

questions_json = {
    "ischemia_risk": {
        "type": "choice",
        "instructions": "Classify the ischemia risk level based on telemetry.",
        "criteria": {
            "NORMAL": "st_elevation < 1.0mm, upright T-wave",
            "WARNING": "st_elevation between 1.0mm and 2.0mm, inverted T-wave",
            "CRITICAL": "st_elevation > 2.0mm, marked ST elevation"
        }
    },
    "st_t_anomaly": {
        "type": "noul",
        "instructions": "Is an ST-T wave anomaly present?"
    }
}

result_2 = agent.predict(state_json, questions_json)
print("Telemetry State:", state_json)
print("Ischemia Risk:", result_2["answers"]["ischemia_risk"]["choice"])
print("Probabilities:", result_2["answers"]["ischemia_risk"]["probabilities"])
print("ST-T Anomaly Probability:", result_2["answers"]["st_t_anomaly"]["noul"])
print()

