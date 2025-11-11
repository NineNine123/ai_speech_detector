import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import glob

def get_latest_model(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def combine_models():
    # Find most recent models
    text_model_path = get_latest_model("models/text_model_*.pkl")
    text_scaler_path = get_latest_model("models/text_scaler_*.pkl")
    acoustic_model_path = get_latest_model("models/acoustic_model_*.pkl")
    acoustic_scaler_path = get_latest_model("models/acoustic_scaler_*.pkl")

    if not all([text_model_path, acoustic_model_path]):
        print("❌ No trained models found.")
        return

    print(f"🧩 Using:\n{text_model_path}\n{acoustic_model_path}")

    # Load models + scalers
    text_model = joblib.load(text_model_path)
    text_scaler = joblib.load(text_scaler_path)
    acoustic_model = joblib.load(acoustic_model_path)
    acoustic_scaler = joblib.load(acoustic_scaler_path)

    # Load features + metadata
    text_features = pd.read_csv("features/text_features.csv")
    audio_features = np.load("features/audio_features.npy")
    meta = pd.read_csv("data/metadata.csv")
    y_true = meta["label"].map({"human": 1, "ai": 0}).values

    # Scale features
    text_scaled = text_scaler.transform(text_features)
    audio_scaled = acoustic_scaler.transform(audio_features)

    # Get probabilities
    text_probs = text_model.predict_proba(text_scaled)[:, 1]
    acoustic_probs = acoustic_model.predict_proba(audio_scaled)[:, 1]

    # Weighted combination (you can tune weights)
    combined_probs = 0.3 * text_probs + 0.7 * acoustic_probs
    y_pred = (combined_probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    print("🔮 Combined Model Accuracy:", round(acc, 3))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    combine_models()
