from pathlib import Path
from typing import List, Optional
from mmce.estimators.base_cardinality_estimator import (
    BatchCardinalityEstimator,
    CardinalityEstimator,
)
import torch
import math
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage


class ClusterGaussianEstimator(CardinalityEstimator):
    """
    Fits an isotropic Gaussian distribution to each cluster in the embedding space.
    Calculates soft probabilities using the survival function of the Normal distribution.
    """

    def __init__(self):
        self.cluster_weights = None
        self.cluster_means = None
        self.cluster_stds = None
        self.embed_dim = 0

    def num_skip(self):
        return 100

    def fit(
        self, images: List[Path], image_embeddings: torch.Tensor, n_components: int
    ):
        self.embed_dim = image_embeddings.shape[1]

        # Fast clustering
        kmeans = MiniBatchKMeans(
            n_clusters=n_components, random_state=42, n_init="auto"
        )
        labels = torch.tensor(
            kmeans.fit_predict(image_embeddings.cpu().numpy()),
            device=image_embeddings.device,
        )

        self.cluster_means = torch.zeros(
            (n_components, self.embed_dim), device=image_embeddings.device
        )
        self.cluster_weights = torch.zeros(n_components, device=image_embeddings.device)
        self.cluster_stds = torch.zeros(n_components, device=image_embeddings.device)

        for i in range(n_components):
            mask = labels == i
            cluster_pts = image_embeddings[mask]
            weight = cluster_pts.shape[0]
            self.cluster_weights[i] = weight

            if weight > 0:
                # 1. Unnormalized mean
                mu = cluster_pts.mean(dim=0)
                self.cluster_means[i] = mu

                # 2. Variance approximation: (1 - ||mu||^2) / D
                mu_norm_sq = (mu**2).sum().item()
                var = max(0.0, 1.0 - mu_norm_sq) / self.embed_dim
                self.cluster_stds[i] = math.sqrt(var)

        # Filter empty clusters
        valid = self.cluster_weights > 0
        self.cluster_means = self.cluster_means[valid]
        self.cluster_weights = self.cluster_weights[valid]

        # Clamp std to avoid division by zero in CDF when variance is exactly 0 (e.g., clusters of size 1)
        self.cluster_stds = torch.clamp(self.cluster_stds[valid], min=1e-8)

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        assert predicate_embedding is not None, (
            "ClusterGaussianEstimator requires predicate_embedding to be not None."
        )
        if (
            self.cluster_means is None
            or self.cluster_stds is None
            or self.cluster_weights is None
        ):
            raise ValueError("Run fit first!")

        # Expected dot product for each cluster
        mean_sims = self.cluster_means @ predicate_embedding

        # Distribution of dot products
        dist = torch.distributions.Normal(mean_sims, self.cluster_stds)

        # Probability mass above the threshold: P(X >= threshold) = 1 - CDF(threshold)
        probs = 1.0 - dist.cdf(
            torch.tensor(threshold, device=predicate_embedding.device)
        )

        # Expected cardinality is the sum of (probability * cluster_size)
        expected_count = (probs * self.cluster_weights).sum()
        return expected_count.item()


class BatchClusterGaussianEstimator(BatchCardinalityEstimator):
    def fit_estimate(
        self,
        images: List[Path],
        predicates: List[str],
        image_embeddings: torch.Tensor,
        predicate_embedding: torch.Tensor,
        threshold: float,
    ) -> List[float]:
        N, D = image_embeddings.shape
        device = image_embeddings.device

        if N == 1:
            sim = (image_embeddings[0] @ predicate_embedding).item()
            return [1.0 if sim >= threshold else 0.0]

        # 1. Compute full hierarchy once (O(N^2))
        # metric='cosine' computes 1 - cosine_similarity, which is correct for distance
        Z = linkage(image_embeddings.cpu().numpy(), metric="cosine", method="average")

        # 2. Pre-allocate tracking arrays for all 2N-1 nodes in the tree
        sum_vecs = torch.zeros((2 * N - 1, D), device=device)
        weights = torch.zeros(2 * N - 1, device=device)

        # Base leaves (the original embeddings)
        sum_vecs[:N] = image_embeddings
        weights[:N] = 1.0

        # Build the intermediate nodes
        for i in range(N - 1):
            idx1, idx2 = int(Z[i, 0]), int(Z[i, 1])
            sum_vecs[N + i] = sum_vecs[idx1] + sum_vecs[idx2]
            weights[N + i] = weights[idx1] + weights[idx2]

        # 3. Compute estimates for ALL nodes simultaneously using vectorization
        mus = sum_vecs / weights.unsqueeze(1)
        mu_norm_sq = (mus**2).sum(dim=1)

        vars = torch.clamp(1.0 - mu_norm_sq, min=0.0) / D
        stds = torch.clamp(torch.sqrt(vars), min=1e-8)

        mean_sims = mus @ predicate_embedding
        dists = torch.distributions.Normal(mean_sims, stds)
        probs = 1.0 - dists.cdf(torch.tensor(threshold, device=device))

        cluster_estimates = probs * weights

        # 4. Traverse the merge history to get estimates for K = 1...N
        results = [0.0] * N

        # Start at K = N (All individual images)
        current_estimate = cluster_estimates[:N].sum().item()
        results[N - 1] = current_estimate

        # Walk up the tree, simulating the merges
        for i in range(N - 1):
            K = N - i - 1  # The number of clusters after this merge
            idx1, idx2 = int(Z[i, 0]), int(Z[i, 1])
            new_idx = N + i

            # Subtract children, add parent
            current_estimate -= cluster_estimates[idx1].item()
            current_estimate -= cluster_estimates[idx2].item()
            current_estimate += cluster_estimates[new_idx].item()

            results[K - 1] = current_estimate

        return results
