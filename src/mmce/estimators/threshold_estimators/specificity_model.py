from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch
from mmce.estimators.base_threshold_estimator import ThresholdEstimator
from mmce.models.threshold.model import ThresholdModel


class SpecificityModelThresholdEstimator(ThresholdEstimator):
    def __init__(self, specificity_model: ThresholdModel):
        self.specificity_model = specificity_model
        self.specificity_model.eval()
        super().__init__()

    def name(self):
        return type(self).__name__

    def num_skip(self) -> int:
        return 1

    def get_bucket_sizes(self, bucket_sizes: Dict[str, List[int]]) -> List[int]:
        return bucket_sizes["num_embeddings"]

    def is_deterministic(self) -> bool:
        """
        Whether the estimator is deterministic. If False, the estimator may return different estimates on different runs with the same input.
        This is used for evaluation purposes, e.g. to decide whether to average multiple runs or not.
        """
        return True

    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        pass

    @abstractmethod
    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
    ) -> float:
        assert predicate_embedding is not None, (
            "This estimator requires a predicate embedding"
        )
        thresholds = self.specificity_model(torch.stack([predicate_embedding])).detach()
        return thresholds[0].item()
