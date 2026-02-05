import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs.csv"
INTERACTIONS_FILE = DATA_DIR / "interactions.csv"


class ContentBasedRecommender:
    def __init__(self):
        self.jobs = pd.read_csv(JOBS_FILE)
        self.interactions = pd.read_csv(INTERACTIONS_FILE)

        # Combine text fields
        self.jobs["text"] = (
            self.jobs["title"].fillna("") + " " +
            self.jobs["description"].fillna("") + " " +
            self.jobs["category"].fillna("") + " " +
            self.jobs["location"].fillna("")
        )

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )
        self.job_vectors = self.vectorizer.fit_transform(self.jobs["text"])

        # Map job_id → index
        self.job_id_to_idx = {
            job_id: idx for idx, job_id in enumerate(self.jobs["job_id"])
        }

    def get_recommendations(self, user_id, train_interactions, k=10):
        """
        Recommend jobs for a user based on content similarity.
        """
        user_history = train_interactions[
            (train_interactions["user_id"] == user_id) &
            (train_interactions["event_type"].isin(["click", "apply"]))
        ]["job_id"].unique()


        # Cold start → return empty (caller will fall back to baseline)
        if len(user_history) == 0:
            return []

        # Build user profile = average of interacted job vectors
        indices = [
            self.job_id_to_idx[job_id]
            for job_id in user_history
            if job_id in self.job_id_to_idx
        ]

        if not indices:
            return []

        user_vector = np.asarray(self.job_vectors[indices].mean(axis=0))


        similarities = cosine_similarity(user_vector, self.job_vectors).flatten()

        # Exclude already seen jobs
        seen = set(user_history)
        ranked_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for idx in ranked_indices:
            job_id = self.jobs.iloc[idx]["job_id"]
            if job_id not in seen:
                recommendations.append(job_id)
            if len(recommendations) == k:
                break

        return recommendations
