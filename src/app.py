import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "mood_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
STATS_PATH = MODELS_DIR / "feature_stats.json"

# ---------- Load artifacts ----------
st.set_page_config(page_title="🎵 Song Mood Classifier", page_icon="🎶", layout="centered")
st.title("🎵 Song Mood Classifier")
st.caption("Interactive Streamlit app — enter audio features to predict if a track feels **Happy** or **Sad**.")

missing = [p for p in [MODEL_PATH, SCALER_PATH, STATS_PATH] if not p.exists()]
if missing:
    st.error(
        "Model/scaler/stats not found. Train first:\n\n"
        "`uv run python src/train_model.py`"
    )
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(STATS_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES = meta["features"]
STATS = meta["stats"]

# ---------- Sidebar inputs ----------
st.sidebar.header("Input Audio Features")
user_vals = {}
for f in FEATURES:
    fmin = float(STATS[f]["min"])
    fmax = float(STATS[f]["max"])
    fmean = float(STATS[f]["mean"])

    # Sensible defaults and ranges
    if f in ("energy", "danceability", "acousticness", "instrumentalness", "liveness", "valence"):
        # bounded [0..1]
        user_vals[f] = st.sidebar.slider(f, 0.0, 1.0, min(max(fmean, 0.0), 1.0), 0.01)
    elif f == "loudness":
        # typically [-60, 0]
        lo = max(-60.0, fmin)
        hi = min(0.0, fmax)
        default = float(np.clip(fmean, lo, hi))
        user_vals[f] = st.sidebar.slider(f, lo, hi, default, 0.1)
    elif f == "tempo":
        # Typical range ~ [50, 220]
        lo = max(40.0, fmin)
        hi = min(240.0, fmax)
        default = float(np.clip(fmean, lo, hi))
        user_vals[f] = st.sidebar.slider(f, lo, hi, default, 1.0)
    else:
        # fallback numeric
        default = float(np.clip(fmean, fmin, fmax))
        user_vals[f] = st.sidebar.slider(f, float(fmin), float(fmax), default)

col1, col2 = st.columns([1,1])
with col1:
    st.markdown("### Current Inputs")
    st.write({k: round(float(v), 4) for k, v in user_vals.items()})

predict = st.sidebar.button("🎯 Predict Mood")

# ---------- Prediction ----------
if predict:
    x = np.array([[user_vals[f] for f in FEATURES]], dtype=float)
    x_scaled = scaler.transform(x)

    # prediction + confidence (if available)
    pred = model.predict(x_scaled)[0]
    try:
        proba = model.predict_proba(x_scaled)[0]
        conf = float(np.max(proba)) * 100.0
        conf_txt = f" — Confidence: **{conf:.1f}%**"
    except Exception:
        conf_txt = ""

    mood = "Happy" if pred == "Happy" else "Sad"
    emoji = "😊" if mood == "Happy" else "😢"

    st.markdown(f"## Predicted Mood: **{mood}** {emoji}{conf_txt}")

    # fun feedback
    (st.balloons() if mood == "Happy" else st.snow())

    # ---------- Visuals ----------
    st.markdown("### Visualizations")

    # 1) Energy vs Valence (intuitive 2D view)
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Valence (0 sad → 1 happy)")
    ax1.set_ylabel("Energy (0 calm → 1 energetic)")
    ax1.set_title("Energy vs Valence")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.scatter(float(user_vals["valence"]), float(user_vals["energy"]), s=200)
    st.pyplot(fig1)

    # 2) Simple radar plot across all features
    import numpy as np
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    vals = [float(user_vals[f]) for f in FEATURES]
    vals += vals[:1]
    angles += angles[:1]

    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax2.plot(angles, vals, linewidth=2)
    ax2.fill(angles, vals, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]), FEATURES)
    ax2.set_title("Feature Profile")
    st.pyplot(fig2)

st.markdown(
    """
**Notes**
- Labels are derived from `valence` (≥0.6 → Happy, ≤0.4 → Sad).
- Add features like `speechiness` or `danceability` weighting to experiment.
- Want more classes (Chill, Energetic, …)? Extend labeling and retrain!
"""
)
