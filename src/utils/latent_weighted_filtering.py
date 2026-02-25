"""Sampling-based latent filtering built on density-aware weighting."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .dataset_filters import DatasetFilter, DatasetFilterContext, DatasetSamplingPlan
from .latent_weighting import LatentDensityKNNWeighter


class LatentWeightedFiltering(DatasetFilter[Any]):
    """Sample source indices from density-derived weights.

    This wrapper delegates weight computation/cache handling to
    ``LatentDensityKNNWeighter`` and then performs weighted sampling from the
    resulting probability distribution.

    Returned plan uses unit weights (all ones) to avoid double counting; the
    weighting effect is encoded in the sampled index frequencies.
    """

    def __init__(
        self,
        num_samples: int,
        sampling_seed: int = 0,
        replacement: bool = True,
        density_weighter: Optional[LatentDensityKNNWeighter] = None,
        **density_weighter_kwargs: Any,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        if density_weighter is not None and density_weighter_kwargs:
            raise ValueError(
                "Provide either density_weighter or density_weighter_kwargs, not both."
            )

        self.num_samples = int(num_samples)
        self.sampling_seed = int(sampling_seed)
        self.replacement = bool(replacement)
        self.density_weighter = (
            density_weighter
            if density_weighter is not None
            else LatentDensityKNNWeighter(**density_weighter_kwargs)
        )

    def build_plan(self, context: DatasetFilterContext[Any]) -> DatasetSamplingPlan:
        weighted_plan = self.density_weighter.build_plan(context)

        selected_indices = weighted_plan.selected_indices.detach().to(
            dtype=torch.long,
            device="cpu",
        )
        weights = weighted_plan.weights.detach().to(dtype=torch.float32, device="cpu")

        if selected_indices.ndim != 1 or weights.ndim != 1:
            raise ValueError("Weighted plan must contain 1D indices and 1D weights.")
        if selected_indices.numel() != weights.numel():
            raise ValueError(
                "Weighted plan size mismatch: "
                f"{selected_indices.numel()} indices vs {weights.numel()} weights."
            )
        if selected_indices.numel() == 0:
            raise ValueError("Cannot sample from an empty weighted plan.")
        if not torch.isfinite(weights).all():
            raise ValueError("Weighted plan contains non-finite weights.")
        if (weights < 0).any():
            raise ValueError("Weighted plan contains negative weights.")

        total_weight = float(weights.sum().item())
        if total_weight <= 0:
            raise ValueError("Weighted plan weights must have positive sum.")

        if not self.replacement and self.num_samples > selected_indices.numel():
            raise ValueError(
                "num_samples exceeds available samples with replacement=False: "
                f"num_samples={self.num_samples}, available={selected_indices.numel()}."
            )

        probabilities = weights / total_weight
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sampling_seed)
        sampled_positions = torch.multinomial(
            probabilities,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=generator,
        )
        sampled_indices = selected_indices[sampled_positions]

        return DatasetSamplingPlan(
            selected_indices=sampled_indices,
            weights=torch.ones(self.num_samples, dtype=torch.float32),
        )
