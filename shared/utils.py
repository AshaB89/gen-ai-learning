"""
Simple shared utilities for AI/ML projects.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace and special characters
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def normalize_data(data):
    """Normalize numerical data using z-score normalization."""
    if isinstance(data, pd.DataFrame):
        return (data - data.mean()) / data.std()
    else:
        return (data - np.mean(data)) / np.std(data)
