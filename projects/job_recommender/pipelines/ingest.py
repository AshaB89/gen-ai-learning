import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("projects/job-recommender/data/raw")
PROCESSED_DATA_PATH = Path("projects/job-recommender/data/processed")

RAW_FILE = RAW_DATA_PATH / "jobs.csv"   # change if needed
OUTPUT_FILE = PROCESSED_DATA_PATH / "jobs.csv"


def ingest_jobs():
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    # Select & rename required columns
    df = df.rename(columns={
        "jobtitle": "title",
        "jobdescription": "description",
        "joblocation_address": "location",
        "industry": "category"
    })

    required_cols = [
        "title",
        "description",
        "company",
        "location",
        "category",
        "uniq_id"
    ]

    df = df[required_cols]

    # Drop rows missing critical information
    df = df.dropna(subset=["title", "description"])

    # Create internal stable job_id
    df = df.reset_index(drop=True)
    df["job_id"] = df.index.astype(int)

    # Reorder columns
    df = df[
        ["job_id", "title", "description", "company", "location", "category", "uniq_id"]
    ]

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned jobs to {OUTPUT_FILE}")
    print(f"Total jobs: {len(df)}")


if __name__ == "__main__":
    ingest_jobs()