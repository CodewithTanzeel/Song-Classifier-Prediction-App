"""
Train models on the Kaggle Spotify Tracks dataset and save:
- models/mood_model.pkl  (best of LR vs RF)
- models/scaler.pkl      (StandardScaler fit on training data)
- models/feature_stats.json (for Streamlit slider ranges)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocess import load_and_prepare_data, compute_feature_stats

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "dataset.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "mood_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
STATS_PATH = MODELS_DIR / "feature_stats.json"

RANDOM_STATE = 42

def evaluate(name, model, X_eval, y_true):
    pred = model.predict(X_eval)
    acc = accuracy_score(y_true, pred)
    print(f"\n--- {name} ---")
    print("Accuracy:", f"{acc:.4f}")
    print(classification_report(y_true, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))
    return acc

def main():
    # 1) Load + label
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_CSV}. "
            "Download Kaggle CSV and place it there as 'dataset.csv'."
        )

    df, FEATURES = load_and_prepare_data(str(DATA_CSV))
    print("Data after filtering:", df.shape, "Happy ratio:", (df['mood'] == 'Happy').mean())

    # 2) Split
    X = df[FEATURES].copy()
    y = df["mood"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3) Scale (kept separate for clarity)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Train two models
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    # RF is not sensitive to scaling, but we’ll keep inputs consistent:
    rf.fit(X_train_scaled, y_train)

    # 5) Evaluate
    acc_lr = evaluate("Logistic Regression", lr, X_test_scaled, y_test)
    acc_rf = evaluate("Random Forest", rf, X_test_scaled, y_test)

    best_model = rf if acc_rf >= acc_lr else lr
    best_name = "Random Forest" if best_model is rf else "Logistic Regression"
    print(f"\nSelected best model: {best_name}")

    # 6) Save model + scaler + feature stats
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    stats = compute_feature_stats(df, FEATURES)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "stats": stats}, f, indent=2)

    print(f"\nSaved:")
    print(f"- Model:  {MODEL_PATH}")
    print(f"- Scaler: {SCALER_PATH}")
    print(f"- Stats:  {STATS_PATH}")

if __name__ == "__main__":
    main()
