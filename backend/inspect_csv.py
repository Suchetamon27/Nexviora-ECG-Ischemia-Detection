import os
import pandas as pd

POSSIBLE_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "ecg_session_log.csv")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ecg_session_log.csv")),
]

CSV_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    CSV_PATH = POSSIBLE_PATHS[0]

def inspect_log():
    if not os.path.exists(CSV_PATH):
        print(f"No CSV log file found at {CSV_PATH}. Make sure the backend has started!")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        print("=" * 80)
        print(f"   NEXVIORA ROBUST PREPROCESSING & AI EVALUATION LOG INSPECTOR")
        print(f"   CSV File: {CSV_PATH}")
        print("=" * 80)
        print(f"Total Recorded Chunks: {len(df)}")
        
        if len(df) == 0:
            print("CSV is currently empty (waiting for data chunks).")
            return

        # Quality Status Breakdown
        status_col = 'quality_status' if 'quality_status' in df.columns else 'pass1_status'
        status_counts = df[status_col].value_counts().to_dict()
        print(f"\n--- Signal Quality Status Breakdown ({status_col}) ---")
        for status, count in status_counts.items():
            pct = (count / len(df)) * 100
            print(f" - {status}: {count} chunks ({pct:.1f}%)")

        # Artifact incidence
        if 'detected_artifacts' in df.columns:
            all_artifacts = []
            for item in df['detected_artifacts'].dropna():
                if item and item != "NONE":
                    all_artifacts.extend([a.strip() for a in item.split(";") if a.strip()])
            if all_artifacts:
                art_series = pd.Series(all_artifacts).value_counts().to_dict()
                print("\n--- Detected Artifact Incidence ---")
                for art, cnt in art_series.items():
                    print(f" - {art}: {cnt} occurrences")

        clean_mask = df[status_col].isin(['CLEAN', 'CLEAN_APPROVED', 'DEGRADED', 'DEGRADED_APPROVED'])
        clean_df = df[clean_mask]
        print(f"\n--- Chunks Passed to Laya AI ({len(clean_df)} of {len(df)}) ---")
        
        if len(clean_df) > 0:
            class_counts = clean_df['classification'].value_counts().to_dict()
            for cls, count in class_counts.items():
                print(f" - {cls}: {count} chunks")
            
            clean_df['ischemia_probability'] = pd.to_numeric(clean_df['ischemia_probability'], errors='coerce')
            avg_prob = clean_df['ischemia_probability'].mean()
            max_prob = clean_df['ischemia_probability'].max()
            min_prob = clean_df['ischemia_probability'].min()
            
            print(f"\nIschemia Probability Stats:")
            print(f" - Mean Risk: {avg_prob * 100:.1f}%")
            print(f" - Min Risk:  {min_prob * 100:.1f}%")
            print(f" - Max Risk:  {max_prob * 100:.1f}%")
            
            print("\nClinical Feature Summary Across Evaluated Chunks:")
            print(f" - Mean ST Elevation:  {clean_df['st_elevation_mm'].mean():.2f} mm")
            print(f" - Mean ST Depression: {clean_df['st_depression_mm'].mean():.2f} mm")

            if 'quality_score' in clean_df.columns:
                print(f" - Mean Signal Quality: {clean_df['quality_score'].mean():.2f} / 1.00")

            # Show latest prompt and raw response sample
            latest_clean = clean_df.iloc[-1]
            print("\n--- Sample AI Prompt & Model Response (Latest Evaluated Chunk) ---")
            print(f"Chunk ID: {latest_clean['chunk_id']}")
            print(f"System Prompt:\n{str(latest_clean.get('ai_system_prompt', 'N/A'))[:250]}...")
            print(f"Input State JSON:\n{latest_clean.get('ai_input_state_json', 'N/A')}")
            print(f"Raw Model Response JSON:\n{latest_clean.get('ai_raw_response_json', 'N/A')}")

        print("\n" + "=" * 80)
        print("Latest 5 Chunks Summary Table:")
        cols = ['chunk_id', status_col, 'quality_score', 'snr_db', 'ischemia_probability', 'classification']
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].tail(5).to_string(index=False))
        print("=" * 80)

    except Exception as e:
        print(f"Error inspecting CSV log: {e}")

if __name__ == "__main__":
    inspect_log()
