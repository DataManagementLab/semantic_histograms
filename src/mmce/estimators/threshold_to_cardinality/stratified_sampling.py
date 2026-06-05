from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

from mmce.estimators.base_cardinality_estimator import CardinalityEstimator
import numpy as np
import torch
from typing import List, Optional


class StratifiedSamplingEstimator(CardinalityEstimator):
    """
    Cluster embeddings using MiniBatchKMeans into K clusters (K <= n_components).
    Then allocate a small sample budget to each cluster (proportional or uniform)
    and use per-cluster sample fractions to estimate counts.
    n_components controls the total sample budget (not number of clusters).
    """

    def __init__(self, n_init_kmeans: int = 3, seed: int = 42):
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.cluster_samples = {}  # cluster_id -> torch.Tensor (samples)
        self.cluster_sizes = None  # np.array of sizes
        self.dataset_size = 0
        self.total_sample_budget = 0
        self.n_init_kmeans = n_init_kmeans
        self.seed = seed
        self.n_clusters = 0

    def num_skip(self) -> int:
        return 100

    def fit(
        self, images: List[Path], image_embeddings: torch.Tensor, n_components: int
    ):
        """
        Here: choose number of clusters k such that k = min(n_images, max(1, n_components // s))
        where s is per-cluster minimal sample (we will set s ~ 4).
        But to keep it simple: set n_clusters = min(n_images, n_components).
        Then allocate ~1 sample per cluster and distribute remainder proportionally to cluster size.
        """
        X = image_embeddings.cpu().numpy()
        n = X.shape[0]
        if n == 0:
            raise ValueError("Empty dataset")

        # special-case: if n_components >= n -> keep entire dataset clustered as singletons: use exact
        if n_components >= n:
            # store full dataset as a single cluster sample
            self.kmeans = None
            self.cluster_samples = {"all": image_embeddings.clone()}
            self.cluster_sizes = np.array([n], dtype=int)
            self.dataset_size = n
            self.total_sample_budget = n_components
            self.n_clusters = 1
            return

        # choose number of clusters
        # A reasonable choice is k = min(n_components, n)
        k = int(min(n_components, n))
        # but limit k not to be huge; user gave n_components to control complexity
        k = max(1, k)
        self.n_clusters = k

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.seed,
            n_init=self.n_init_kmeans,  # type: ignore
        )
        labels = km.fit_predict(X)
        self.kmeans = km
        self.dataset_size = n
        # cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        sizes = np.zeros(k, dtype=int)
        sizes[unique] = counts
        self.cluster_sizes = sizes

        # allocate samples per cluster: proportional to cluster sizes but ensure at least 1 per cluster when possible
        budget = n_components
        per_cluster = np.maximum(1, (budget * sizes / sizes.sum()).astype(int))
        # adjust to exactly budget
        diff = budget - per_cluster.sum()
        # distribute remaining
        i = 0
        while diff > 0:
            per_cluster[i % k] += 1
            i += 1
            diff -= 1
        while diff < 0:
            # remove from largest cluster with >1
            idx = np.argmax(per_cluster)
            if per_cluster[idx] > 1:
                per_cluster[idx] -= 1
                diff += 1
            else:
                break

        # sample inside each cluster
        self.cluster_samples = {}
        X_torch = torch.from_numpy(X).float()
        for c in range(k):
            members_idx = np.where(labels == c)[0]
            m = len(members_idx)
            if m == 0:
                self.cluster_samples[c] = torch.empty((0, X_torch.shape[1]))
                continue
            s = min(per_cluster[c], m)
            # choose random without replacement
            inds = np.random.default_rng(self.seed + c).choice(m, size=s, replace=False)
            chosen_idx = members_idx[inds]
            self.cluster_samples[c] = X_torch[chosen_idx]

        self.total_sample_budget = n_components

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        assert predicate_embedding is not None, (
            "StratifiedSamplingEstimator requires predicate_embedding to be not None."
        )
        if self.cluster_samples is None:
            raise ValueError("Run fit first!")
        p = predicate_embedding.cpu()
        # if single 'all' cluster
        if isinstance(self.cluster_samples, dict) and list(
            self.cluster_samples.keys()
        ) == ["all"]:
            sims = (self.cluster_samples["all"] @ p).cpu().numpy()
            frac = float((sims >= threshold).sum()) / len(sims)
            return frac * self.dataset_size

        # otherwise aggregate per cluster
        assert self.cluster_sizes is not None
        est = 0.0
        for c, sample in self.cluster_samples.items():
            if isinstance(c, str):
                continue
            if sample.numel() == 0:
                continue
            sims = (sample @ p).cpu().numpy()
            # fraction in sample exceeding threshold
            frac = float((sims >= threshold).sum()) / len(sims)
            est += frac * float(self.cluster_sizes[int(c)])
        return est
