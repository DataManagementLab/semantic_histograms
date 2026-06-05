from pathlib import Path
from typing import List, Optional
from mmce.estimators.base_cardinality_estimator import (
    BatchCardinalityEstimator,
    CardinalityEstimator,
)
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage


class CoresetEstimator(CardinalityEstimator):
    """
    Stratified sampling using normalized K-Means centroids weighted by cluster size.
    """

    def __init__(self):
        self.core_points = None
        self.cluster_weights = None

    def num_skip(self):
        return 100

    def fit(
        self, images: List[Path], image_embeddings: torch.Tensor, n_components: int
    ):
        kmeans = MiniBatchKMeans(
            n_clusters=n_components, random_state=42, n_init="auto"
        )
        labels = torch.tensor(
            kmeans.fit_predict(image_embeddings.cpu().numpy()),
            device=image_embeddings.device,
        )

        # Get centroids and normalize them back to the hypersphere
        centroids = torch.tensor(
            kmeans.cluster_centers_, device=image_embeddings.device
        )
        self.core_points = torch.nn.functional.normalize(centroids, p=2, dim=1)

        self.cluster_weights = torch.zeros(n_components, device=image_embeddings.device)
        for i in range(n_components):
            self.cluster_weights[i] = (labels == i).sum()

        valid = self.cluster_weights > 0
        self.core_points = self.core_points[valid]
        self.cluster_weights = self.cluster_weights[valid]

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        assert predicate_embedding is not None, (
            "CoresetEstimator requires predicate_embedding to be not None."
        )
        if self.core_points is None or self.cluster_weights is None:
            raise ValueError("Run fit first!")

        similarities = self.core_points @ predicate_embedding
        matches = (similarities >= threshold).float()

        total_count = (matches * self.cluster_weights).sum()
        return total_count.item()


class BatchCoresetEstimator(BatchCardinalityEstimator):
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

        Z = linkage(image_embeddings.cpu().numpy(), metric="cosine", method="average")

        sum_vecs = torch.zeros((2 * N - 1, D), device=device)
        weights = torch.zeros(2 * N - 1, device=device)

        sum_vecs[:N] = image_embeddings
        weights[:N] = 1.0

        for i in range(N - 1):
            idx1, idx2 = int(Z[i, 0]), int(Z[i, 1])
            sum_vecs[N + i] = sum_vecs[idx1] + sum_vecs[idx2]
            weights[N + i] = weights[idx1] + weights[idx2]

        # For Coreset, we treat the normalized centroid as the sample
        mus = sum_vecs / weights.unsqueeze(1)
        norms = torch.norm(mus, p=2, dim=1, keepdim=True)
        normalized_mus = mus / torch.clamp(norms, min=1e-8)

        sims = normalized_mus @ predicate_embedding
        matches = (sims >= threshold).float()

        cluster_estimates = matches * weights

        results = [0.0] * N
        current_estimate = cluster_estimates[:N].sum().item()
        results[N - 1] = current_estimate

        for i in range(N - 1):
            K = N - i - 1
            idx1, idx2 = int(Z[i, 0]), int(Z[i, 1])
            new_idx = N + i

            current_estimate -= cluster_estimates[idx1].item()
            current_estimate -= cluster_estimates[idx2].item()
            current_estimate += cluster_estimates[new_idx].item()

            results[K - 1] = current_estimate

        return results
