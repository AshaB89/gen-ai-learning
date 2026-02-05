from evaluation.data_loader import load_interactions
from evaluation.split import time_based_split
from evaluation.metrics import evaluate_model
from models.content_based import ContentBasedRecommender
from models.baseline import get_top_k_jobs


def main():
    interactions = load_interactions()
    train, test = time_based_split(interactions)

    model = ContentBasedRecommender()

    def content_recommender(user_id, k=10):
        recs = model.get_recommendations(user_id, train, k)
        # Cold-start fallback
        if not recs:
            return get_top_k_jobs(k)
        return recs

    metrics = evaluate_model(
        get_recommendations_fn=content_recommender,
        test_interactions=test,
        k=10
    )

    print("Content-based results:")
    print(metrics)


if __name__ == "__main__":
    main()
