import numpy as np


def recall_at_k(recommended, relevant, k):
    """
    Recall@K: did we retrieve at least one relevant item?
    """
    recommended_k = recommended[:k]
    return int(len(set(recommended_k) & set(relevant)) > 0)


def mrr_at_k(recommended, relevant, k):
    """
    Mean Reciprocal Rank@K.
    """
    for rank, job_id in enumerate(recommended[:k], start=1):
        if job_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_model(get_recommendations_fn, test_interactions, k=10):
    """
    Evaluate a recommender function using Recall@K and MRR@K.

    Parameters
    ----------
    get_recommendations_fn : callable
        Function(user_id, k) -> list of job_ids
    test_interactions : pd.DataFrame
        Test interactions (user_id, job_id)
    k : int
        Top-K recommendations

    Returns
    -------
    dict with Recall@K and MRR@K
    """
    recalls = []
    mrrs = []

    for user_id, group in test_interactions.groupby("user_id"):
        relevant_jobs = group["job_id"].tolist()
        recommended_jobs = get_recommendations_fn(user_id, k)

        recalls.append(recall_at_k(recommended_jobs, relevant_jobs, k))
        mrrs.append(mrr_at_k(recommended_jobs, relevant_jobs, k))

    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"MRR@{k}": float(np.mean(mrrs)),
    }
