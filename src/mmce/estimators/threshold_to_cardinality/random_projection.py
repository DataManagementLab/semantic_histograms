from pathlib import Path
import torch
from typing import List, Optional

from mmce.estimators.base_cardinality_estimator import (
    BatchCardinalityEstimator,
    CardinalityEstimator,
)


class RandomProjectionEstimator(CardinalityEstimator):
    """
    Approximates similarities by projecting embeddings onto a random orthogonal subspace.
    The subspace dimension K is scaled to match the memory footprint of storing n_components.
    """

    def __init__(self):
        self.sketched_embeddings = None
        self.projection_matrix = None
        self.K = 0

    def fit(
        self, images: List[Path], image_embeddings: torch.Tensor, n_components: int
    ):
        N, D = image_embeddings.shape
        device = image_embeddings.device

        # Fair memory allocation: K * N == n_components * D
        self.K = max(1, (n_components * D) // N)
        self.K = min(self.K, D)  # Cap at D to avoid going out of bounds

        # Generate K random orthogonal directions (Haar measure)
        gen = torch.Generator(device=device).manual_seed(42)
        random_matrix = torch.randn(D, D, device=device, generator=gen)
        Q, _ = torch.linalg.qr(random_matrix)
        self.projection_matrix = Q[:, : self.K]

        # Project and re-normalize
        P_X = image_embeddings @ self.projection_matrix
        norms = torch.norm(P_X, p=2, dim=1, keepdim=True)
        self.sketched_embeddings = P_X / torch.clamp(norms, min=1e-8)

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        if self.sketched_embeddings is None or self.projection_matrix is None:
            raise ValueError("Run fit first!")

        # Project and normalize the query
        P_q = predicate_embedding @ self.projection_matrix
        norm_q = torch.norm(P_q, p=2)
        normalized_q = P_q / torch.clamp(norm_q, min=1e-8)

        # Estimate cardinality
        sims = self.sketched_embeddings @ normalized_q
        matches = (sims >= threshold).sum()
        return matches.item()


class BatchRandomProjectionEstimator(BatchCardinalityEstimator):
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

        # 1. Generate the full orthogonal rotation matrix once
        gen = torch.Generator(device=device).manual_seed(42)
        random_matrix = torch.randn(D, D, device=device, generator=gen)
        Q, _ = torch.linalg.qr(random_matrix)

        # 2. Project everything into the new orthogonal basis
        P_X = image_embeddings @ Q
        P_q = predicate_embedding @ Q

        # 3. Compute cumulative dot products and norms across dimensions 1 to D
        cum_dot = torch.cumsum(P_X * P_q, dim=1)  # Shape: (N, D)
        cum_norm_X = torch.sqrt(torch.cumsum(P_X**2, dim=1))  # Shape: (N, D)
        cum_norm_q = torch.sqrt(torch.cumsum(P_q**2, dim=0))  # Shape: (D,)

        cum_norm_X = torch.clamp(cum_norm_X, min=1e-8)
        cum_norm_q = torch.clamp(cum_norm_q, min=1e-8)

        # 4. Approximate cosine similarities for all projection dimensions K
        approx_sims = cum_dot / (cum_norm_X * cum_norm_q)  # Shape: (N, D)

        # Pre-calculate how many matches exist for every K
        matches_per_K = (approx_sims >= threshold).sum(dim=0).float()  # Shape: (D,)

        # 5. Map n_components -> K based on our fairness multiplier
        results = []
        for i in range(1, N + 1):
            K = max(1, (i * D) // N)

            # K is 1-indexed, so we access index K-1 (capping at D-1)
            K_idx = min(K, D) - 1
            results.append(matches_per_K[K_idx].item())

        return results
