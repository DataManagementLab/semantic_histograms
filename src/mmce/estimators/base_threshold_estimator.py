from abc import abstractmethod
from typing import Dict, List, Literal, Optional
from pathlib import Path
import torch

from mmce.estimators.base_cardinality_estimator import CardinalityEstimator


class ThresholdEstimator:
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
        Fit the ThresholdEstimator to the available data.
        Args:
        - images: The image paths to fit. Guranteed to be in the same order as the image_embeddings, but guaranteed to be in the same order as the dataset used for evaluation.
        - image_embeddings: The image embeddings to fit. Guranteed to be normalized. Shape: (n_images, embed_dim).
        - n_components: The number of components of the estimator. In the range [1, n_images].
          Could be: number of buckets, clustes, base distributions to fit, samples, ...
          The estimator should be more accurate the higher n_components, but more expensive to run.
          Often the estimator collapses to comparing all image_embeddings with the query embedding when n_components==n_images.
          Feel free to multily the value by a constant if the underlying is not naturally in the range [1, n_images],
          e.g. if image embeddings are projected onto a number of random directions.
        - seed: Seed for rng.
        """

    @abstractmethod
    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
    ) -> float:
        """
        Run the threshold estimation. `fit` must be called first.
        Returns an estimate of the similarity threshold.
        Args:
        - predicate: The filter predicate as a string. This is only used for non-embedding-based methods, and can be ignored otherwise.
        - predicate_embedding: The embedding of the filter predicate. Guranteed to be normalized. Shape: (embed_dim,).
        Returns:
        - threshold: The similarity threshold. Return an estimate of the similarity threshold to use for cardinality estimation.
        """
        pass


class ThresholdToCardinality:
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
        Fit the ThresholdToCardinality to the available data.
        Args:
        - images: The image paths to fit. Guranteed to be in the same order as the image_embeddings, but guaranteed to be in the same order as the dataset used for evaluation.
        - image_embeddings: The image embeddings to fit. Guranteed to be normalized. Shape: (n_images, embed_dim).
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
        threshold: float,
    ) -> float:
        """
        Run the cardinality estimation. `fit` must be called first.
        Returns an estimate on the number of image_embeddings that have higher cosine similarity than the threshold.
        Args:
        - predicate: The filter predicate as a string. This is only used for non-embedding-based methods, and can be ignored otherwise.
        - predicate_embedding: The embedding of the filter predicate. Guranteed to be normalized. Shape: (embed_dim,).
        - threshold: The similarity threshold. Return an estimate of the number of images that have higher
          cosine similarity to the predicate embedding than this threshold.
        """
        pass


class ThresholdBasedCardinalityEstimator(CardinalityEstimator):
    def __init__(
        self,
        threshold_estimator: ThresholdEstimator,
        threshold_to_cardinality: ThresholdToCardinality,
        determines_bucket_sizes: Literal[
            "threshold_estimator", "threshold_to_cardinality"
        ],
        other_bucket_size: int,
    ):
        self.threshold_estimator = threshold_estimator
        self.threshold_to_cardinality = threshold_to_cardinality
        self.other_bucket_size = other_bucket_size
        self.determines_bucket_sizes = determines_bucket_sizes

    def name(self):
        return (
            f"{self.threshold_estimator.name()}-{self.threshold_to_cardinality.name()}"
        )

    def num_skip(self) -> int:
        return max(
            self.threshold_estimator.num_skip(),
            self.threshold_to_cardinality.num_skip(),
        )

    def embedding_based(self) -> bool:
        return True

    def get_bucket_sizes(self, bucket_sizes: Dict[str, List[int]]) -> List[int]:
        if self.determines_bucket_sizes == "threshold_estimator":
            return self.threshold_estimator.get_bucket_sizes(bucket_sizes)
        else:
            return self.threshold_to_cardinality.get_bucket_sizes(bucket_sizes)

    def is_deterministic(self) -> bool:
        """
        Whether the estimator is deterministic. If False, the estimator may return different estimates on different runs with the same input.
        This is used for evaluation purposes, e.g. to decide whether to average multiple runs or not.
        """
        return (
            self.threshold_estimator.is_deterministic()
            and self.threshold_to_cardinality.is_deterministic()
        )

    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        if self.determines_bucket_sizes == "threshold_estimator":
            self.threshold_estimator.fit(
                images=images,
                image_embeddings=image_embeddings,
                n_components=n_components,
                seed=seed,
            )
            self.threshold_to_cardinality.fit(
                images=images,
                image_embeddings=image_embeddings,
                n_components=self.other_bucket_size,
                seed=seed,
            )
        else:
            self.threshold_estimator.fit(
                images=images,
                image_embeddings=image_embeddings,
                n_components=self.other_bucket_size,
                seed=seed,
            )
            self.threshold_to_cardinality.fit(
                images=images,
                image_embeddings=image_embeddings,
                n_components=n_components,
                seed=seed,
            )

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
    ) -> float:
        threshold = self.threshold_estimator.estimate(predicate, predicate_embedding)
        cardinality = self.threshold_to_cardinality.estimate(
            predicate, predicate_embedding, threshold
        )
        return cardinality


class SamplingEstimator(ThresholdToCardinality):
    def __init__(self):
        self.n_components = 0
        self.dataset_size = 0
        self.image_embeddings = None

    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        assert n_components <= len(image_embeddings), (
            "n_components must be less than or equal to the number of image embeddings."
        )
        self.image_embeddings = image_embeddings
        self.n_components = n_components
        self.dataset_size = len(image_embeddings)
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        assert predicate_embedding is not None, (
            "SamplingEstimator requires predicate_embedding to be not None."
        )
        if self.image_embeddings is None:
            raise ValueError("Must call fit before estimate.")
        idx = torch.multinomial(
            torch.ones(self.dataset_size),
            num_samples=self.n_components,
            generator=self.gen,
            replacement=False,
        )
        sample = self.image_embeddings[idx]
        similarities = sample @ predicate_embedding.unsqueeze(1)
        sample_count = (similarities >= threshold).sum()
        total_count = (sample_count / self.n_components) * self.dataset_size
        return total_count.item()


class FullEstimator(ThresholdToCardinality):
    def __init__(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(42)
        self.n_components = 0
        self.dataset_size = 0
        self.image_embeddings = None

    def is_deterministic(self) -> bool:
        return True

    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        self.image_embeddings = image_embeddings
        self.dataset_size = len(image_embeddings)

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
        threshold: float,
    ) -> float:
        assert predicate_embedding is not None, (
            "FullEstimator requires predicate_embedding to be not None."
        )
        if self.image_embeddings is None:
            raise ValueError("Must call fit before estimate.")
        similarities = self.image_embeddings @ predicate_embedding.unsqueeze(1)
        count = (similarities >= threshold).sum()
        return count.item()


class ComboThresholdEstimator(ThresholdEstimator):
    def __init__(
        self,
        primary: ThresholdEstimator,
        secondary: ThresholdEstimator,
        secondary_n_components: int,
    ):
        self.primary = primary
        self.secondary = secondary
        self.secondary_n_components = secondary_n_components

    def name(self):
        return f"combo-{self.primary.name()}-{self.secondary.name()}"

    def num_skip(self) -> int:
        return max(self.primary.num_skip(), self.secondary.num_skip())

    def get_bucket_sizes(self, bucket_sizes: Dict[str, List[int]]) -> List[int]:
        return self.primary.get_bucket_sizes(bucket_sizes)

    def is_deterministic(self) -> bool:
        """
        Whether the estimator is deterministic. If False, the estimator may return different estimates on different runs with the same input.
        This is used for evaluation purposes, e.g. to decide whether to average multiple runs or not.
        """
        return self.primary.is_deterministic() and self.secondary.is_deterministic()

    @abstractmethod
    def fit(
        self,
        images: List[Path],
        image_embeddings: torch.Tensor,
        n_components: int,
        seed: int,
    ):
        self.primary.fit(
            images=images,
            image_embeddings=image_embeddings,
            n_components=n_components,
            seed=seed,
        )
        self.secondary.fit(
            images=images,
            image_embeddings=image_embeddings,
            n_components=self.secondary_n_components,
            seed=seed,
        )

    @abstractmethod
    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[torch.Tensor],
    ) -> float:
        primary_threshold = self.primary.estimate(predicate, predicate_embedding)
        secondary_threshold = self.secondary.estimate(predicate, predicate_embedding)
        # For simplicity, we just average the two thresholds. More complex combinations are possible.
        return (primary_threshold + secondary_threshold) / 2
