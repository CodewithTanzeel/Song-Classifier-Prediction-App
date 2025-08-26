import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mood_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
STATS_PATH = ROOT / "models" / "feature_stats.json"

# ---------- Load artifacts ----------
st.set_page_config(page_title="🎵 Song Mood Classifier", page_icon="🎶", layout="wide")
st.title("🎵 Song Mood Classifier")
st.caption("Predict whether a song feels **Happy** or **Sad** from audio features.")

# Missing model check
for p in [MODEL_PATH, SCALER_PATH, STATS_PATH]:
    if not p.exists():
        st.error("Missing model/scaler/stats. Train first: `uv run python src/train_model.py`")
        st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(STATS_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES = meta["features"]
STATS = meta["stats"]

# ---------- Sidebar inputs ----------
st.sidebar.header("🎚 Input Audio Features")
user_vals = {}
for f in FEATURES:
    lo, hi, mean = float(STATS[f]["min"]), float(STATS[f]["max"]), float(STATS[f]["mean"])
    default = np.clip(mean, lo, hi)
    step = 0.01 if 0 <= lo < 1 and hi <= 1 else 1.0
    user_vals[f] = st.sidebar.slider(f, lo, hi, float(default), step=step)

# ---------- Prediction ----------
if st.sidebar.button("🎯 Predict Mood"):
    X = np.array([[user_vals[f] for f in FEATURES]], dtype=float)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    mood = "Happy" if pred == "Happy" else "Sad"
    emoji = "😊" if mood == "Happy" else "😢"

    try:
        conf = model.predict_proba(X_scaled).max() * 100
        conf_txt = f" — Confidence: **{conf:.1f}%**"
    except:
        conf_txt = ""

    st.subheader(f"Predicted Mood: **{mood}** {emoji}{conf_txt}")
    st.balloons() if mood == "Happy" else st.snow()

    # ---------- Graphical Insights ----------
    col1, col2, col3 = st.columns(3)

    # 1) Energy vs Valence scatter
    with col1:
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Valence (Happiness)")
        ax.set_ylabel("Energy")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.scatter(user_vals["valence"], user_vals["energy"], s=200, color="orange")
        ax.set_title("Energy vs Valence")
        st.pyplot(fig)

    # 2) Radar chart
    with col2:
        angles = np.linspace(0, 2*np.pi, len(FEATURES), endpoint=False).tolist()
        vals = [user_vals[f] for f in FEATURES]
        vals += vals[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, vals, "o-", linewidth=2)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), FEATURES)
        ax.set_title("Feature Profile")
        st.pyplot(fig)

    # 3) Histogram of input features
    with col3:
        fig, ax = plt.subplots()
        ax.bar(FEATURES, [user_vals[f] for f in FEATURES], color="skyblue")
        ax.set_xticklabels(FEATURES, rotation=45, ha="right")
        ax.set_title("Feature Levels")
        st.pyplot(fig)

    # ---------- Correlation Heatmap ----------
    st.subheader("📊 Feature Correlation Heatmap")
    df = pd.DataFrame([user_vals])
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---------- Data Table ----------
    st.subheader("📄 Current Feature Values")
    st.write(pd.DataFrame([user_vals]))
