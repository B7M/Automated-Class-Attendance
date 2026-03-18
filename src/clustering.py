# Unsupervised identity clustering and visualization utilities.

import os
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from config import (
    CLUSTERS_DIR,
    CLUSTER_MAPPING_PATH,
    FACE_CROPS_DIR,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    KMEANS_K_RANGE,
)


def plot_elbow(embeddings: np.ndarray, k_range=KMEANS_K_RANGE) -> None:
    """Plot KMeans inertia vs. K to help choose the number of clusters."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(embeddings)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", linestyle="--")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot — Optimal K")
    plt.xticks(list(k_range))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_k_distance(embeddings: np.ndarray, k: int = 4) -> None:
    """Plot sorted k-th nearest-neighbor distances to guide DBSCAN eps selection."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    k_distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(10, 5))
    plt.plot(k_distances, label=f"{k}-NN distance")
    plt.xlabel(f"Points sorted by distance to {k}-th neighbor")
    plt.ylabel("Distance")
    plt.title(f"K-Distance Plot (k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_dbscan(
    embeddings: np.ndarray,
    filenames: list[str],
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    clusters_dir: str = CLUSTERS_DIR,
    face_crops_dir: str = FACE_CROPS_DIR,
    cluster_mapping_path: str = CLUSTER_MAPPING_PATH,
) -> np.ndarray:
    """
    Cluster embeddings with DBSCAN and organize face crops into per-cluster folders.

    Args:
        embeddings:           Array of shape (N, 512).
        filenames:            Corresponding .npy filenames.
        eps, min_samples:     DBSCAN hyperparameters.
        clusters_dir:         Root directory for cluster output folders.
        face_crops_dir:       Source directory for face crop images.
        cluster_mapping_path: Output path for the filename → cluster JSON.

    Returns:
        Array of cluster labels (shape N,). Noise points are labeled -1.
    """
    if os.path.exists(clusters_dir):
        shutil.rmtree(clusters_dir)
    os.makedirs(clusters_dir, exist_ok=True)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = dbscan.fit_predict(embeddings)

    cluster_mapping = {}
    for label, filename in zip(labels, filenames):
        if label == -1:
            continue  # Noise point — skip
        cluster_folder = os.path.join(clusters_dir, f"student_{label}")
        os.makedirs(cluster_folder, exist_ok=True)

        image_file = filename.replace(".npy", ".jpg")
        image_path = os.path.join(face_crops_dir, image_file)
        if os.path.exists(image_path):
            shutil.copy(image_path, cluster_folder)

        cluster_mapping[filename] = int(label)

    with open(cluster_mapping_path, "w") as f:
        json.dump(cluster_mapping, f, indent=4)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points. Mapping saved to '{cluster_mapping_path}'.")
    return labels


def plot_pca(embeddings: np.ndarray, labels: np.ndarray) -> None:
    """Visualize embeddings in 2D using PCA, colored by cluster label."""
    reduced = PCA(n_components=2).fit_transform(embeddings)
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=20)
    plt.title("PCA — Embeddings by Cluster")
    plt.tight_layout()
    plt.show()


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray) -> None:
    """Visualize embeddings in 2D using t-SNE, colored by cluster label."""
    reduced = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(embeddings)
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=20)
    plt.title("t-SNE — Embeddings by Cluster")
    plt.tight_layout()
    plt.show()