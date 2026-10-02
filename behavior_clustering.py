import numpy as np
from sklearn.cluster import KMeans


# Behavioral features used by AI 4
FEATURES = [
    "journal_count",
    "app_open_count",
    "coping_sessions",
    "checkin_completion",
    "sleep_time_shift"
]


def prepare_data(days):
    """
    Convert behavioral records into a numerical matrix.
    """

    return np.array([
        [float(day.get(feature, 0)) for feature in FEATURES]
        for day in days
    ])


def elbow_method(days, max_k=5):
    """
    Elbow Method:
    Calculates K-Means inertia for different values of K.
    """

    data = prepare_data(days)

    if len(data) < 2:
        return {
            "status": "insufficient_data",
            "message": "At least 2 behavioral records are required."
        }

    # K cannot be greater than number of records
    max_k = min(max_k, len(data))

    results = []

    for k in range(1, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(data)

        results.append({
            "k": k,
            "inertia": float(model.inertia_)
        })

    return {
        "status": "success",
        "algorithm": "Elbow Method",
        "elbow_values": results
    }


def cluster_behavior(days, n_clusters=2):
    """
    K-Means Clustering:
    Groups behavioral records into clusters.
    """

    data = prepare_data(days)

    if len(data) < n_clusters:
        return {
            "status": "insufficient_data",
            "message": f"At least {n_clusters} behavioral records are required."
        }

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(data)

    return {
        "status": "success",
        "algorithm": "K-Means Clustering",
        "n_clusters": n_clusters,
        "labels": labels.tolist(),
        "cluster_centers": model.cluster_centers_.tolist(),
        "inertia": float(model.inertia_)
    }