import pandas as pd
from pathlib import Path

# Resolve project root dynamically
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs.csv"
INTERACTIONS_FILE = DATA_DIR / "interactions.csv"


def load_jobs():
    """
    Load processed jobs data.
    """
    return pd.read_csv(JOBS_FILE)


def load_interactions():
    """
    Load processed user–job interaction data.
    """
    df = pd.read_csv(INTERACTIONS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
