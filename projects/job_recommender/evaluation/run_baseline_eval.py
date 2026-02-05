from evaluation.data_loader import load_interactions
from evaluation.split import time_based_split
from evaluation.metrics import evaluate_model
from models.baseline import get_top_k_jobs


# Adapter: evaluation expects (user_id, k) -> list[job_id]
def baseline_recommender(user_id, k=10):
    return get_top_k_jobs(k)


def main():
    # 1. Load interaction data
    interactions = load_interactions()

    # 2. Split into train / test by time
    train, test = time_based_split(interactions)

    # 3. Evaluate baseline recommender
    metrics = evaluate_model(
        get_recommendations_fn=baseline_recommender,
        test_interactions=test,
        k=10
    )

    print("Baseline results:")
    print(metrics)


if __name__ == "__main__":
    main()
