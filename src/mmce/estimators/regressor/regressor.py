from typing import List, Optional
from mmce.models.regressor.model import MatchClusterModel
from mmce.estimators.base_cardinality_estimator import (
    BatchCardinalityEstimator,
    CardinalityEstimator,
)
import torch
from safetensors.torch import load_file
from pathlib import Path


class RegressorEstimatorBatch(BatchCardinalityEstimator):
    def __init__(self, model_path=Path("artifacts/embeddings/model.safetensors")):
        self.device = torch.device("cuda:0")
        state_dict = load_file(model_path)
        self.model = MatchClusterModel.from_state_dict(state_dict).to(self.device)
        self.model.eval()

    def fit_estimate(
        self,
        images: List[Path],
        predicates: List[str],
        image_embeddings: torch.Tensor,
        predicate_embedding: torch.Tensor,
        threshold: float,
    ) -> List[float]:
        return (
            self.model.compute_cardinalities(image_embeddings, predicate_embedding)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )


class RegressorEstimator(CardinalityEstimator):
    def __init__(self, model_path=Path("artifacts/embeddings/model.safetensors")):
        self.device = torch.device("cuda:0")
        state_dict = load_file(model_path)
        self.model = MatchClusterModel.from_state_dict(state_dict).to(self.device)
        self.model.eval()

    def fit(
        self, images: List[Path], image_embeddings: torch.Tensor, n_components: int
    ):
        alg = self.model.fit_clustering(image_embeddings)
        children = torch.from_numpy(alg.children_).int()
        cluster_elements = self.model.get_cluster_elements(alg)
        cluster_embeddings, cluster_features = (
            MatchClusterModel.compute_all_cluster_embeddings_and_features(
                image_embeddings, cluster_elements
            )
        )
        self.cluster_sizes = torch.tensor(
            [len(x) for x in cluster_elements], dtype=torch.int
        )
        self.mask = self.model.get_mask_for_num_buckets(
            children, num_buckets=n_components
        )  # Shape: depth x (batch_size, num_clusters)
        num_samples = len(image_embeddings)
        self.max_num_clusters = num_samples * 2 - 1
        self.cluster_embeddings_masked = cluster_embeddings[self.mask].to(self.device)
        self.cluster_features_masked = cluster_features[self.mask].to(self.device)
        self.bucket_sizes = self.cluster_sizes[self.mask].to(self.device)

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ):
        assert predicate_embedding is not None, (
            "RegressorEstimator requires predicate_embedding to be not None."
        )
        output = self.model.forward_single(
            cluster_embeddings=self.cluster_embeddings_masked,
            cluster_features=self.cluster_features_masked,
            query_embeddings=predicate_embedding.expand_as(
                self.cluster_embeddings_masked
            ),
        ).view(-1)
        cluster_cardinality = (output * self.bucket_sizes).sum()
        return float(cluster_cardinality.detach().cpu())
