"""Abstract interfaces for dataset-level sample filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch

TFilterBatch = TypeVar("TFilterBatch")


@dataclass(frozen=True)
class DatasetSamplingPlan:
    """Selected source indices and aligned per-sample weights."""

    selected_indices: torch.Tensor
    weights: torch.Tensor


@dataclass
class DatasetFilterContext(Generic[TFilterBatch]):
    """Context provided to dataset filters during construction-time selection.

    Attributes:
        dataset_name: Human-readable dataset identifier.
        source_length: Number of source samples before filtering.
        sample_shape: Optional per-sample tensor shape expected by the dataset.
        metadata: Dataset-specific immutable metadata useful for filter decisions.
        get_source_samples: Callback that returns deterministic sample batches.
        default_batch_size: Suggested batch size for vectorized filtering.
        state: Mutable dict for sharing prepared state between filter hooks.
    """

    dataset_name: str
    source_length: int
    sample_shape: Optional[Tuple[int, ...]]
    metadata: Mapping[str, Any]
    get_source_samples: Callable[[torch.Tensor], TFilterBatch]
    default_batch_size: int = 1024
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_length < 0:
            raise ValueError("source_length must be >= 0")
        if self.default_batch_size < 1:
            raise ValueError("default_batch_size must be >= 1")

    def normalize_indices(
        self,
        source_indices: Union[Sequence[int], torch.Tensor],
    ) -> torch.Tensor:
        """Normalize indices to a 1D CPU LongTensor."""
        if isinstance(source_indices, torch.Tensor):
            if source_indices.ndim != 1:
                raise ValueError(
                    f"source_indices tensor must be 1D, got {source_indices.ndim}D"
                )
            return source_indices.detach().to(dtype=torch.long, device="cpu")
        return torch.as_tensor(list(source_indices), dtype=torch.long)

    def get_source_sample(self, source_idx: int) -> TFilterBatch:
        """Fetch one sample via the batched retrieval API."""
        index = torch.tensor([int(source_idx)], dtype=torch.long)
        return self.get_source_samples(index)

    def iter_index_batches(
        self,
        batch_size: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield contiguous source index batches for streaming filters."""
        batch = self.default_batch_size if batch_size is None else int(batch_size)
        if batch < 1:
            raise ValueError("batch_size must be >= 1")

        for start in range(0, self.source_length, batch):
            end = min(start + batch, self.source_length)
            yield torch.arange(start, end, dtype=torch.long)


class DatasetFilter(ABC, Generic[TFilterBatch]):
    """Abstract base class for dataset-level sample selection policies."""

    def prepare(self, context: DatasetFilterContext[TFilterBatch]) -> None:
        """Optional global pre-computation stage before index selection."""

    @abstractmethod
    def build_plan(
        self,
        context: DatasetFilterContext[TFilterBatch],
    ) -> DatasetSamplingPlan:
        """Return selected source indices and aligned sample weights."""


class PerSampleDatasetFilter(DatasetFilter[TFilterBatch], ABC):
    """Helper base class for vectorized per-sample filtering policies."""

    def __init__(self, batch_size: Optional[int] = None) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def keep_batch(
        self,
        batch: TFilterBatch,
        source_indices: torch.Tensor,
        context: DatasetFilterContext[TFilterBatch],
    ) -> torch.Tensor:
        """Return a 1D boolean keep mask aligned with source_indices."""

    def weight_batch(
        self,
        batch: TFilterBatch,
        source_indices: torch.Tensor,
        context: DatasetFilterContext[TFilterBatch],
    ) -> torch.Tensor:
        """Return a 1D float tensor of sample weights aligned with source_indices."""
        return torch.ones(source_indices.numel(), dtype=torch.float32)

    def build_plan(
        self,
        context: DatasetFilterContext[TFilterBatch],
    ) -> DatasetSamplingPlan:
        batch_size = context.default_batch_size if self.batch_size is None else self.batch_size
        selected_batches: list[torch.Tensor] = []
        weight_batches: list[torch.Tensor] = []

        for index_batch in context.iter_index_batches(batch_size=batch_size):
            batch = context.get_source_samples(index_batch)
            keep_mask = self.keep_batch(batch, index_batch, context)
            if keep_mask.ndim != 1:
                raise ValueError(
                    f"keep_batch must return a 1D tensor, got {keep_mask.ndim}D"
                )
            if keep_mask.numel() != index_batch.numel():
                raise ValueError(
                    "keep_batch output size mismatch: "
                    f"expected {index_batch.numel()}, got {keep_mask.numel()}"
                )
            keep_mask = keep_mask.to(dtype=torch.bool, device=index_batch.device)

            batch_weights = self.weight_batch(batch, index_batch, context)
            if batch_weights.ndim != 1:
                raise ValueError(
                    f"weight_batch must return a 1D tensor, got {batch_weights.ndim}D"
                )
            if batch_weights.numel() != index_batch.numel():
                raise ValueError(
                    "weight_batch output size mismatch: "
                    f"expected {index_batch.numel()}, got {batch_weights.numel()}"
                )
            batch_weights = batch_weights.to(dtype=torch.float32, device=index_batch.device)

            kept = index_batch[keep_mask]
            if kept.numel() > 0:
                selected_batches.append(kept.cpu())
                weight_batches.append(batch_weights[keep_mask].cpu())

        if not selected_batches:
            return DatasetSamplingPlan(
                selected_indices=torch.empty(0, dtype=torch.long),
                weights=torch.empty(0, dtype=torch.float32),
            )

        return DatasetSamplingPlan(
            selected_indices=torch.cat(selected_batches, dim=0),
            weights=torch.cat(weight_batches, dim=0),
        )
