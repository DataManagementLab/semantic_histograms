from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path
import torch


class CardinalityEstimator(ABC):
    def name(self):
        return type(self).__name__

    def num_skip(self) -> int:
        return 1

    def embedding_based(self) -> bool:
        return False

    def get_bucket_sizes(self, bucket_sizes: Dict[str, List[int]]) -> List[int]:
        return bucket_sizes["num_embeddings"]

    def is_deterministic(self) -> bool:
        """
        Whether the estimator is deterministic. If False, the estimator may return different estimates on different runs with the same input.
        This is used for evaluation purposes, e.g. to decide whether to average multiple runs or not.
        """
        return False

    @abstractmethod
    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        """
        Fit the CardinalityEstimate to the available data.

        Args:
        - images: The image paths to fit. Guranteed to be in the same order as the image_embeddings, but guaranteed to be in the same order as the dataset used for evaluation.
        - image_embeddings: The image embeddings to fit. Guraranteed to be normalized. Shape: (n_images, embed_dim).
        - n_components: The number of components of the estimator. In the range [1, n_images].
          Could be: number of buckets, clustes, base distributions to fit, samples, ...
          The estimator should be more accurate the higher n_components, but more expensive to run.
          Often the estimator collapses to comparing all image_embeddings with the query embedding when n_components==n_images.
          Feel free to multily the value by a constant if the underlying is not naturally in the range [1, n_images],
          e.g. if image embeddings are projected onto a number of random directions.
        """

    @abstractmethod
    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
    ) -> float:
        """
        Run the cardinality estimation. `fit` must be called first.
        Returns an estimate on the number of image_embeddings that have higher cosine similarity than the threshold.

        Args:
        - predicate: The filter predicate as a string. This is only used for non-embedding-based methods, and can be ignored otherwise.
        - predicate_embedding: The embedding of the filter predicate. Guraranteed to be normalized. Shape: (embed_dim,).
        - threshold: The similarity threshold. Return an estimate of the number of images that have higher
          cosine similarity to the predicate embedding than this threshold.
        """


# class BatchCardinalityEstimator(ABC):
#     @classmethod
#     def name(cls):
#         return cls.__name__
#
#     @abstractmethod
#     def fit_estimate(
#         self,
#         images: List[Path],
#         predicates: List[str],
#         image_embeddings: torch.Tensor,
#         predicate_embedding: torch.Tensor,
#         threshold: float,
#     ) -> List[float]:
#         """
#         Fit the CardinalityEstimate and directly output all cardinality estimates for all n_components.
#
#         Args:
#         - images: The image paths to fit. Guranteed to be in the same order as the image_embeddings, but guaranteed to be in the same order as the dataset used for evaluation.
#         - predicates: The filter predicates as strings. This is only used for non-embedding-based
#         - image_embeddings: The image embeddings to fit. Guraranteed to be normalized. Shape: (n_images, embed_dim).
#         - predicate_embedding: The embedding of the filter predicate. Guraranteed to be normalized. Shape: (embed_dim,).
#         - threshold: The similarity threshold. Return an estimate of the number of images that have higher
#           cosine similarity to the predicate embedding than this threshold.
#
#         Returns:
#         - Cardinality Estimates for n_components = 1, ..., n_image_embeddings.
#         """


class RandomOrder(CardinalityEstimator):
    def __init__(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(42)

    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        self.dataset_size = len(image_embeddings)

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
    ) -> float:
        x = torch.rand(size=(1,), generator=self.gen)
        total_count = x * self.dataset_size
        return total_count.item()
