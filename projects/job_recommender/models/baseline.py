import pandas as pd
from pathlib import Path

# Resolve project root dynamically
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

INTERACTIONS_FILE = PROCESSED_DATA_PATH / "interactions.csv"
OUTPUT_FILE = PROCESSED_DATA_PATH / "top_jobs.csv"

EVENT_WEIGHTS = {
    "view": 1,
    "click": 3,
    "apply": 5
}


def build_popularity_model(top_k=50):
    """
    OFFLINE STEP:
    Build popularity-based ranking and save to CSV.
    """
    interactions = pd.read_csv(INTERACTIONS_FILE)

    interactions["weight"] = interactions["event_type"].map(EVENT_WEIGHTS)

    popularity = (
        interactions
        .groupby("job_id")["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "popularity_score"})
        .sort_values("popularity_score", ascending=False)
    )

    top_jobs = popularity.head(top_k)
    top_jobs.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved top {top_k} jobs to {OUTPUT_FILE}")


def get_top_k_jobs(k=10):
    """
    ONLINE STEP:
    Return top-K popular jobs for recommendation.
    """
    df = pd.read_csv(OUTPUT_FILE)
    return df["job_id"].head(k).tolist()


if __name__ == "__main__":
    build_popularity_model()
