# metrics.py

import numpy as np

def precision_at_k(y_true, k):
    """Calculate precision at k for recommendation."""
    if len(y_true) == 0:
        return 0
    # Ensure k is not larger than the number of true ratings
    k = min(k, len(y_true))
    top_k_predictions = np.argsort(y_true)[-k:]
    relevant_items = np.sum(y_true[top_k_predictions])
    precision = relevant_items / k
    return precision

def recall_at_k(y_true, k, num_relevant):
    """Calculate recall at k for recommendation."""
    if num_relevant == 0:
        return 0
    k = min(k, len(y_true))
    top_k_predictions = np.argsort(y_true)[-k:]
    relevant_items = np.sum(y_true[top_k_predictions])
    recall = relevant_items / num_relevant
    return recall

def F1(precision, recall):
    """Calculate F1 score given precision and recall."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def mean_average_precision(y_true):
    """Calculate mean average precision for recommendation."""
    precision_at_i = [precision_at_k(y_true, i + 1) for i in range(len(y_true))]
    return np.mean(precision_at_i)

def ndcg_at_k(y_true, k):
    """Calculate normalized discounted cumulative gain at k for recommendation."""
    if len(y_true) == 0:
        return 0
    # Discounted cumulative gain
    dcg = 0
    for i in range(min(k, len(y_true))):
        dcg += (2**y_true[i] - 1) / np.log2(i + 2)
    # Ideal DCG (sorted in descending order)
    ideal_dcg = sum((2**rating - 1) / np.log2(i + 2) for i, rating in enumerate(sorted(y_true, reverse=True)))
    # NDCG
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
    return ndcg
