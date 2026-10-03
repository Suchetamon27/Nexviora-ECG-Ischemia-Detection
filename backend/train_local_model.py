import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Import your existing feature extraction
from app.signal.features import extract_clinical_features
from app.signal.filters import clean_5s_chunk

def load_and_extract_features(dataset_path: str):
    """
    Load CSV files, clean the signals, and extract features.
    Assumes your dataset has a folder structure like:
    dataset/
      normal/
        ecg_001.csv
        ecg_002.csv
      ischemia/
        ecg_001.csv
        ...
    """
    X = []
    y = []
    
    # Define labels based on folder names
    labels_map = {"normal": 0, "ischemia": 1, "other": 2}
    
    print("Extracting features from dataset...")
    for label_name, label_val in labels_map.items():
        folder_path = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder_path):
            continue
            
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file in csv_files:
            try:
                # Assuming the CSV has a column 'ecg'
                df = pd.read_csv(file)
                raw_signal = df['ecg'].values
                
                # We need exactly 5 seconds (1250 samples at 250Hz) or chunk it
                # For simplicity, assume the CSV is roughly a 5s chunk
                if len(raw_signal) < 1250:
                    # Pad if too short
                    raw_signal = np.pad(raw_signal, (0, 1250 - len(raw_signal)), 'edge')
                else:
                    raw_signal = raw_signal[:1250]
                
                # 1. Clean the signal
                is_clean, noise_pct, clean_sig = clean_5s_chunk(raw_signal, sample_rate=250)
                
                if not is_clean:
                    print(f"Skipping {file} due to high noise ({noise_pct:.1f}%)")
                    continue
                
                # 2. Extract Features
                features = extract_clinical_features(clean_sig, sample_rate=250)
                
                if features:
                    # Flatten features into an array
                    feature_vector = [
                        features["heart_rate"],
                        features["st_deviation_uv"],
                        features["t_wave_amplitude_uv"],
                        1 if features["wellens_warning"] else 0,
                        1 if features["hyperacute_t_warning"] else 0
                    ]
                    
                    X.append(feature_vector)
                    y.append(label_val)
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
    return np.array(X), np.array(y)

def train():
    # 1. Prepare Data
    # NOTE: Change this path to where your ~1000 CSV files are stored!
    DATASET_DIR = "dataset/" 
    
    if not os.path.exists(DATASET_DIR):
        print(f"Please create a '{DATASET_DIR}' folder with 'normal' and 'ischemia' subfolders containing your CSVs.")
        return

    X, y = load_and_extract_features(DATASET_DIR)
    
    if len(X) == 0:
        print("No valid data found or processed. Exiting.")
        return
        
    print(f"Successfully extracted features for {len(X)} samples.")
    
    # 2. Split into Train / Test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and Train the Model (Random Forest is great for tabular features and fast on CPU)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Ischemia", "Other"][:len(np.unique(y))]))
    
    # 5. Save the Model
    model_path = "ecg_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to {model_path}!")
    print("You can now load this model in your worker.py using joblib.load('ecg_classifier.pkl') instead of calling Laya.")

if __name__ == "__main__":
    train()
