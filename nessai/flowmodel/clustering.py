import logging
import numpy as np
from typing import Any, Tuple

from .base import FlowModel
from ..utils.stats import silhouette_score

logger = logging.getLogger(__name__)


class ClusteringFlowModel(FlowModel):
    """Flow model that includes clustering."""

    max_n_clusters: int = 20
    n_clusters: int = None
    cluster_weights: np.ndarray = None

    def setup_from_input_dict(self, config: dict) -> None:
        super().setup_from_input_dict(config)
        max_n_clusters = config.pop("max_n_clusters", None)
        if max_n_clusters is not None:
            self.max_n_clusters = max_n_clusters
        self.model_config["kwargs"]["context_features"] = 1

    def train_clustering(self, samples: np.ndarray, **kwargs) -> np.ndarray:
        """Train the clustering algorithm using a set of samples.

        Tries :code:`n_clusters=2` up to :code:`max_n_clusters`
        and chooses the number that has highest silhouette score.

        Returns
        -------
        Array of labels for the training samples.
        """
        import faiss

        best_score = -np.inf
        dims = samples.shape[-1]
        for n_clusters in range(2, self.max_n_clusters + 1):
            kmeans = faiss.Kmeans(
                dims, n_clusters, nredo=5, niter=20, **kwargs
            )
            kmeans.train(samples)
            score = np.mean(silhouette_score(samples, kmeans))
            logger.info(f"k={n_clusters}, score={score}")
            if score > best_score:
                best_score = score
                best_clusterer = kmeans

        self.clusterer = best_clusterer
        labels = self.get_cluster_labels(samples)
        unique_labels = np.unique(labels)
        self.n_clusters = len(unique_labels)
        logger.info(f"n_clusters={self.n_clusters}")
        self.cluster_weights = np.bincount(
            labels.flatten(), minlength=self.max_n_clusters
        ) / len(samples)
        logger.info(f"cluster_weights={self.cluster_weights}")
        return labels

    def get_cluster_labels(
        self, samples: np.ndarray, clusterer=None
    ) -> np.ndarray:
        """Get the cluster labels for a set of samples."""
        if clusterer is None:
            clusterer = self.clusterer
        _, labels = clusterer.index.search(samples, 1)
        return labels.reshape(-1, 1)

    def sample_cluster_labels(self, n: int) -> np.ndarray:
        """Sample n random cluster labels"""
        return np.random.choice(
            self.max_n_clusters, size=(n, 1), p=self.cluster_weights
        )

    def train(self, samples: np.ndarray, **kwargs) -> dict:
        """Train the normalising flow and clustering"""
        cluster_labels = self.train_clustering(samples)
        return super().train(samples, conditional=cluster_labels, **kwargs)

    def forward_and_log_prob(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cluster_labels = self.get_cluster_labels(x)
        return super().forward_and_log_prob(x, conditional=cluster_labels)

    def sample(self, n: int = 1, return_labels: bool = False) -> np.ndarray:
        cluster_labels = self.sample_cluster_labels(n)
        samples = super().sample(n, conditional=cluster_labels)
        if return_labels:
            return samples, cluster_labels
        else:
            return samples

    def sample_and_log_prob(
        self, N: int = 1, z: np.ndarray = None, alt_dist: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if z is not None:
            N = len(z)
        cluster_labels = self.sample_cluster_labels(N)
        return super().sample_and_log_prob(
            N=N, z=z, alt_dist=alt_dist, conditional=cluster_labels
        )
