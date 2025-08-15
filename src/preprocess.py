import pandas as pd

# Core feature set we’ll use
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

def load_and_prepare_data(csv_path: str):
    """
    Load the Kaggle Spotify Tracks dataset and create a clean, labeled
    binary classification frame using valence thresholds:
      valence >= 0.6 -> Happy
      valence <= 0.4 -> Sad
    Mid-range rows are dropped to keep labels crisp.
    Returns: df (with 'mood'), features list.
    """
    df = pd.read_csv(csv_path)

    # Keep only the columns we need and drop NaNs
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[FEATURES].dropna()

    # derive labels from valence
    df = df[(df["valence"] >= 0.6) | (df["valence"] <= 0.4)].copy()
    df["mood"] = df["valence"].apply(lambda v: "Happy" if v >= 0.6 else "Sad")

    return df, FEATURES


def compute_feature_stats(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Compute min/max/mean for each feature for nice Streamlit slider ranges.
    """
    stats = {}
    for f in features:
        s = df[f]
        stats[f] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }
    return stats
