import pandas as pd


def time_based_split(interactions, test_ratio=0.2):
    """
    Split interactions per user by time.
    Last `test_ratio` fraction of events goes to test set.

    Parameters
    ----------
    interactions : pd.DataFrame
        Must contain columns: user_id, job_id, timestamp
    test_ratio : float
        Fraction of each user's interactions used for testing

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    interactions = interactions.sort_values("timestamp")

    train_parts = []
    test_parts = []

    for user_id, group in interactions.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))

        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df
