from pathlib import Path
import numpy as np

from mmce.estimators.base_cardinality_estimator import CardinalityEstimator
import math
import torch
from typing import List, Optional
from scipy.stats import norm  # scipy required

from sklearn.mixture import GaussianMixture


class GaussianMixtureEstimator(CardinalityEstimator):
    """
    Fit a GaussianMixture to the embeddings. For a query predicate p and threshold t,
    we compute for each Gaussian component j the 1D normal distribution of the scalar
    p^T X when X ~ N(mu_j, Sigma_j). That distribution is normal with mean p^T mu_j
    and variance p^T Sigma_j p. The probability that p^T X >= t is then
    1 - Phi((t - mean) / sqrt(var)), and we weight by the component weight.
    """

    def __init__(
        self, covariance_type: str = "diag", seed: int = 0, max_iter: int = 200
    ):
        assert covariance_type in ("full", "tied", "diag", "spherical")
        self.covariance_type = covariance_type
        self.gmm: Optional[GaussianMixture] = None
        self.dataset_size = 0
        self.fitted = False
        self.seed = seed
        self.max_iter = max_iter

    def num_skip(self) -> int:
        return 100

    def fit(
        self, images: List[Path], image_embeddings: torch.Tensor, n_components: int
    ):
        X = image_embeddings.cpu().numpy()
        n, d = X.shape
        if n_components >= n:
            # fit GMM with n components is expensive and unnecessary; treat as exact fallback
            # but we can still fit a GMM with n_components = min(n,  min(n, 50)) to avoid explosion
            # For exact fallback, just store embeddings
            self.gmm = None
            self.dataset_size = n
            self._exact_embeddings = X
            self.fitted = True
            return

        # Fit GMM
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.seed,
            max_iter=self.max_iter,
            warm_start=False,
        )
        gm.fit(X)
        self.gmm = gm
        self.dataset_size = n
        self.fitted = True

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        if not self.fitted:
            raise ValueError("Run fit first!")
        assert predicate_embedding is not None, (
            "GaussianMixtureEstimator requires predicate_embedding to be not None."
        )
        # exact fallback
        if self.gmm is None:
            # use exact embeddings
            X = torch.from_numpy(self._exact_embeddings).float()
            sims = (X @ predicate_embedding.cpu()).numpy()
            return float((sims >= threshold).sum())

        p = predicate_embedding.cpu().numpy()  # shape (d,)
        weights = self.gmm.weights_  # (k,)
        means = self.gmm.means_  # (k, d)
        covs = None
        cov_type = self.gmm.covariance_type
        if cov_type == "diag":
            # shape (k, d)
            covs = self.gmm.covariances_
        elif cov_type == "full":
            # shape (k, d, d)
            covs = self.gmm.covariances_
        elif cov_type == "spherical":
            # covariances_ is array (k,) (variance scalar per comp) -> expand
            covs = self.gmm.covariances_
        elif cov_type == "tied":
            covs = self.gmm.covariances_
        else:
            raise NotImplementedError("Unknown covariance type")

        assert weights is not None
        assert means is not None
        assert covs is not None

        # For each component j compute mean_proj and var_proj = p^T Sigma_j p
        total_prob = 0.0
        eps = 1e-12
        k = len(weights)
        # d = p.shape[0]

        for j in range(k):
            mu_j = means[j]  # (d,)
            mean_proj = float(np.dot(p, mu_j))  # scalar
            # compute var_proj depending on covariance representation
            if cov_type == "diag":
                # covs[j] is (d,)
                var_proj = float(np.sum((p**2) * covs[j]))
            elif cov_type == "full":
                Sigma_j = covs[j]  # (d, d)
                var_proj = float(p.dot(Sigma_j.dot(p)))
            elif cov_type == "spherical":
                var_scalar = covs[j]
                var_proj = float(var_scalar * np.sum(p**2))
            elif cov_type == "tied":
                # covs is (d,d)
                var_proj = float(p.dot(covs.dot(p)))
            else:
                var_proj = eps

            if var_proj <= 1e-12:
                # degenerate: treat as point mass
                prob = 1.0 if mean_proj >= threshold else 0.0
            else:
                std = math.sqrt(var_proj)
                z = (threshold - mean_proj) / std
                prob = 1.0 - norm.cdf(z)  # P(proj >= threshold)
            total_prob += weights[j] * prob

        # multiply by dataset size to get count estimate
        est = total_prob * float(self.dataset_size)
        return float(est)
