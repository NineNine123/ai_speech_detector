import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def train_acoustic_model():
    # Load audio features
    X = np.load("features/audio_features.npy")
    df_meta = pd.read_csv("data/metadata.csv")
    y = df_meta["label"].map({"human": 1, "ai": 0}).values

    # Normalize features for neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build MLP model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("🎧 Acoustic Model Accuracy:", round(acc, 3))
    print(classification_report(y_test, y_pred))

    # Save model + scaler with timestamp
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, f"models/acoustic_model_{timestamp}.pkl")
    joblib.dump(scaler, f"models/acoustic_scaler_{timestamp}.pkl")

    print(f"✅ Acoustic model saved as models/acoustic_model_{timestamp}.pkl")

if __name__ == "__main__":
    train_acoustic_model()
