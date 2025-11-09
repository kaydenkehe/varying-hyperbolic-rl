from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class WhiteningStats:
    mean: np.ndarray
    std: np.ndarray


def whiten_features(features: np.ndarray) -> Tuple[np.ndarray, WhiteningStats]:
    """
    Z-score features across the dataset (Section 3 of PLAN).
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    normalized = (features - mean) / std
    return normalized.astype(np.float32), WhiteningStats(mean.squeeze(), std.squeeze())


@dataclass
class ClusteringOutput:
    assignments: np.ndarray
    centers: np.ndarray
    model: MiniBatchKMeans


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    batch_size: int = 8192,
    max_iter: int = 200,
    seed: int = 0,
) -> ClusteringOutput:
    """
    Mini-batch KMeans on latent states to produce graph nodes.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    assignments = kmeans.fit_predict(features)
    return ClusteringOutput(assignments=assignments, centers=kmeans.cluster_centers_, model=kmeans)
