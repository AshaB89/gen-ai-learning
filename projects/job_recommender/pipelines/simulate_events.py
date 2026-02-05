import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROCESSED_DATA_PATH = Path("projects/job-recommender/data/processed")
JOBS_FILE = PROCESSED_DATA_PATH / "jobs.csv"
OUTPUT_FILE = PROCESSED_DATA_PATH / "interactions.csv"

# Simulation parameters
NUM_USERS = 3000
JOBS_PER_USER = (20, 50)   # min, max jobs viewed
CLICK_PROB = 0.05
APPLY_PROB = 0.10


def simulate_interactions():
    jobs = pd.read_csv(JOBS_FILE)
    job_ids = jobs["job_id"].tolist()

    interactions = []
    base_time = datetime.now()

    for user_id in range(NUM_USERS):
        num_jobs = np.random.randint(JOBS_PER_USER[0], JOBS_PER_USER[1])
        sampled_jobs = np.random.choice(job_ids, size=num_jobs, replace=False)

        for job_id in sampled_jobs:
            # VIEW event
            view_time = base_time + timedelta(
                seconds=np.random.randint(0, 100000)
            )
            interactions.append(
                [user_id, job_id, "view", view_time]
            )

            # CLICK event
            if np.random.rand() < CLICK_PROB:
                click_time = view_time + timedelta(
                    seconds=np.random.randint(10, 300)
                )
                interactions.append(
                    [user_id, job_id, "click", click_time]
                )

                # APPLY event
                if np.random.rand() < APPLY_PROB:
                    apply_time = click_time + timedelta(
                        seconds=np.random.randint(60, 1800)
                    )
                    interactions.append(
                        [user_id, job_id, "apply", apply_time]
                    )

    df = pd.DataFrame(
        interactions,
        columns=["user_id", "job_id", "event_type", "timestamp"]
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved interactions to {OUTPUT_FILE}")
    print(f"Total events: {len(df)}")
    print(df["event_type"].value_counts())


if __name__ == "__main__":
    simulate_interactions()
