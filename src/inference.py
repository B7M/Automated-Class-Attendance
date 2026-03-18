# KNN-based identity matching and attendance recording.

import json

import numpy as np
from sklearn.neighbors import NearestNeighbors

from config import STUDENT_MAP_PATH, KNN_NEIGHBORS


def load_student_map(map_path: str = STUDENT_MAP_PATH) -> dict:
    """Load the cluster ID → student name mapping from JSON."""
    with open(map_path, "r") as f:
        return json.load(f)


def match_faces_to_students(
    new_embeddings: np.ndarray,
    enrolled_embeddings: np.ndarray,
    enrolled_labels: np.ndarray,
    student_map: dict,
    n_neighbors: int = KNN_NEIGHBORS,
) -> list[str]:
    """
    Match query embeddings to enrolled identities using KNN.

    Args:
        new_embeddings:      Array of shape (N, 512) from the current session.
        enrolled_embeddings: Array of shape (M, 512) from enrollment.
        enrolled_labels:     Cluster IDs corresponding to enrolled_embeddings.
        student_map:         Mapping of cluster ID (str) → student name.
        n_neighbors:         Number of nearest neighbors to consider.

    Returns:
        List of matched student names (one per query embedding).
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(enrolled_embeddings)

    names = []
    for embedding in new_embeddings:
        _, indices = knn.kneighbors(embedding.reshape(1, -1))
        cluster_id = enrolled_labels[indices[0][0]]
        name = student_map.get(str(cluster_id), "Unknown")
        names.append(name)

    return names


def mark_attendance(present_students: list[str]) -> list[str]:
    """
    Deduplicate and print the set of present students.

    Args:
        present_students: Raw list of matched student names (may contain duplicates).

    Returns:
        Deduplicated list of present student names.
    """
    unique = list(set(present_students))
    print(f"Students present: {unique}")
    return unique