import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def train_text_model():
    df_text = pd.read_csv("features/text_features.csv")
    df_meta = pd.read_csv("data/metadata.csv")

    # Add label
    df_text["label"] = df_meta["label"]
    X = df_text.drop(columns=["label"])
    y = df_text["label"].map({"human": 1, "ai": 0})

    # Normalize features (helps with scale differences)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("🗣️ Text Model Accuracy:", round(acc, 3))
    print(classification_report(y_test, y_pred))

    # Save model + scaler with timestamp
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, f"models/text_model_{timestamp}.pkl")
    joblib.dump(scaler, f"models/text_scaler_{timestamp}.pkl")

    print(f"✅ Text model saved as models/text_model_{timestamp}.pkl")

if __name__ == "__main__":
    train_text_model()
