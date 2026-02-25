"""Common dataset classes for density/susceptibility and latent data."""

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

from ..data.zarr_dataset import NoddyverseZarrDataset
from .dataset_filters import DatasetFilter, DatasetFilterContext, DatasetSamplingPlan

logger = logging.getLogger(__name__)


class JointDensitySuscDataset(Dataset):
    """
    Dataset that returns stacked density and susceptibility tensors.

    Susceptibility is always log10-transformed: log10(susc + 1e-6).
    Density is returned as-is.

    Args:
        zarr_path: Path to the Noddyverse Zarr store.
        return_index: If True, returns (index, tensor) tuple instead of just tensor.
        field: Which raw field(s) to return. One of "both", "density", or "susc".
    """

    def __init__(
        self, zarr_path: str, return_index: bool = False, field: str = "both"
    ) -> None:
        self.base = NoddyverseZarrDataset(
            zarr_path,
            fields=("rock_types",),
            include_metadata=False,
            return_tensors=True,
        )
        self.return_index = return_index
        valid_fields = {"both", "density", "susc"}
        if field not in valid_fields:
            raise ValueError(
                f"field must be one of {sorted(valid_fields)}, got {field!r}"
            )
        self.field = field

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[int, torch.Tensor]]:
        sample = self.base[idx]
        density = sample["density"]
        susceptibility = sample["susceptibility"]
        # Always apply log10 transformation to susceptibility
        susceptibility = torch.log10(susceptibility + 1e-6)
        if self.field == "density":
            stacked = density.unsqueeze(0)
        elif self.field == "susc":
            stacked = susceptibility.unsqueeze(0)
        else:
            stacked = torch.stack([density, susceptibility], dim=0)

        if self.return_index:
            return idx, stacked
        return stacked


@dataclass(frozen=True)
class LatentFilterBatch:
    """Deterministic latent batch payload for dataset filters."""

    source_indices: torch.Tensor
    mu: torch.Tensor
    logvar: Optional[torch.Tensor]


class LatentZarrDataset(Dataset):
    """
    Dataset that samples latents from a Zarr store on each access.

    Supports optional padding to a minimum spatial size and stochastic
    sampling from the VAE posterior when logvar is available.
    Returns (weight, sample) tuples where weights default to 1.0.

    Args:
        zarr_path: Path to the latent Zarr store.
        pad_to: Minimum spatial dimension size (pads with zeros if smaller).
        use_logvar: If True and logvar is available, sample z = mu + exp(0.5*logvar)*eps.
        sample_filter: Optional dataset filter run once at construction time to
            select source sample indices.
    """

    def __init__(
        self,
        zarr_path: str,
        pad_to: int = 0,
        use_logvar: bool = True,
        sample_filter: Optional[DatasetFilter[LatentFilterBatch]] = None,
    ) -> None:
        self.zarr_path = zarr_path
        self.pad_to = max(pad_to, 1)

        group = zarr.open_group(zarr_path, mode="r")
        if "latent_mu" not in group:
            raise ValueError(f"Zarr store missing 'latent_mu': {zarr_path}")

        mu_store = cast(Any, group["latent_mu"])
        self.source_length = int(mu_store.shape[0])
        self.length = self.source_length
        self.channels = int(mu_store.shape[1])
        self.original_spatial: Tuple[int, int, int] = (
            int(mu_store.shape[2]),
            int(mu_store.shape[3]),
            int(mu_store.shape[4]),
        )
        self.padded_spatial = self._compute_padded_spatial(self.original_spatial)
        self.pad_amounts = tuple(
            padded - orig
            for padded, orig in zip(self.padded_spatial, self.original_spatial)
        )
        self.latent_shape = (self.channels,) + self.padded_spatial
        self.has_logvar = "latent_logvar" in group
        self.use_logvar = use_logvar and self.has_logvar
        self.selected_source_indices: Optional[Tuple[int, ...]] = None
        self.selected_weights: Optional[torch.Tensor] = None
        self._default_weight = torch.tensor(1.0, dtype=torch.float32)
        chunks = getattr(mu_store, "chunks", None)
        if chunks is None:
            self.default_filter_batch_size = 1024
        else:
            self.default_filter_batch_size = max(1, int(chunks[0]))

        if use_logvar and not self.has_logvar:
            logger.warning(
                "Latent logvar not found in %s; sampling deterministically from mu.",
                zarr_path,
            )

        self._group: Optional[zarr.Group] = None
        self._mu_store: Optional[Any] = None
        self._logvar_store: Optional[Any] = None

        if sample_filter is not None:
            self._apply_sample_filter(sample_filter)

    def _compute_padded_spatial(
        self, spatial: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        pad_to = self.pad_to
        d, h, w = spatial
        return (max(d, pad_to), max(h, pad_to), max(w, pad_to))

    def _ensure_open(self) -> None:
        if self._group is None:
            group = zarr.open_group(self.zarr_path, mode="r")
            self._group = group
            self._mu_store = group["latent_mu"]
            self._logvar_store = (
                group["latent_logvar"] if "latent_logvar" in group else None
            )

    def _build_filter_context(self) -> DatasetFilterContext[LatentFilterBatch]:
        metadata: Mapping[str, Any] = {
            "zarr_path": self.zarr_path,
            "channels": self.channels,
            "has_logvar": self.has_logvar,
            "use_logvar": self.use_logvar,
            "original_spatial": self.original_spatial,
            "padded_spatial": self.padded_spatial,
            "pad_amounts": self.pad_amounts,
            "latent_shape": self.latent_shape,
            "default_filter_batch_size": self.default_filter_batch_size,
        }
        return DatasetFilterContext(
            dataset_name=self.__class__.__name__,
            source_length=self.source_length,
            sample_shape=self.latent_shape,
            metadata=metadata,
            get_source_samples=self._get_filter_samples,
            default_batch_size=self.default_filter_batch_size,
        )

    def _get_filter_samples(self, source_indices: torch.Tensor) -> LatentFilterBatch:
        self._ensure_open()
        if self._mu_store is None:
            raise RuntimeError("Latent Zarr store not initialized.")

        source_indices = source_indices.detach().to(dtype=torch.long, device="cpu")
        if source_indices.ndim != 1:
            raise ValueError(
                "source_indices must be a 1D tensor, " f"got {source_indices.ndim}D"
            )

        if source_indices.numel() == 0:
            empty_shape = (0,) + self.latent_shape
            return LatentFilterBatch(
                source_indices=source_indices,
                mu=torch.empty(empty_shape, dtype=torch.float32),
                logvar=(
                    torch.empty(empty_shape, dtype=torch.float32)
                    if self._logvar_store is not None
                    else None
                ),
            )

        min_idx = int(torch.min(source_indices).item())
        max_idx = int(torch.max(source_indices).item())
        if min_idx < 0 or max_idx >= self.source_length:
            raise ValueError(
                "Requested filter indices out of bounds for latent source length "
                f"{self.source_length}: min={min_idx}, max={max_idx}."
            )

        index_list = source_indices.tolist()
        mu = torch.as_tensor(self._mu_store[index_list], dtype=torch.float32)
        mu = self._pad_tensor(mu)

        logvar: Optional[torch.Tensor] = None
        if self._logvar_store is not None:
            logvar = torch.as_tensor(
                self._logvar_store[index_list],
                dtype=torch.float32,
            )
            logvar = self._pad_tensor(logvar)

        return LatentFilterBatch(source_indices=source_indices, mu=mu, logvar=logvar)

    def _validate_sampling_plan(
        self,
        plan: DatasetSamplingPlan,
    ) -> Tuple[Tuple[int, ...], torch.Tensor]:
        normalized_indices = plan.selected_indices.detach().to(
            dtype=torch.long, device="cpu"
        )
        normalized_weights = plan.weights.detach().to(dtype=torch.float32, device="cpu")

        if normalized_indices.ndim != 1:
            raise ValueError(
                "Sampling plan selected_indices must be a 1D tensor, "
                f"got {normalized_indices.ndim}D"
            )
        if normalized_weights.ndim != 1:
            raise ValueError(
                "Sampling plan weights must be a 1D tensor, "
                f"got {normalized_weights.ndim}D"
            )
        if normalized_indices.numel() != normalized_weights.numel():
            raise ValueError(
                "Sampling plan size mismatch: "
                f"{normalized_indices.numel()} selected indices but "
                f"{normalized_weights.numel()} weights."
            )

        normalized = tuple(int(i) for i in normalized_indices.tolist())
        if len(normalized) == 0:
            logger.warning("Sample filter selected zero samples for %s", self.zarr_path)
            return normalized, normalized_weights

        min_idx = min(normalized)
        max_idx = max(normalized)
        if min_idx < 0 or max_idx >= self.source_length:
            raise ValueError(
                "Filtered indices out of bounds for latent source length "
                f"{self.source_length}: min={min_idx}, max={max_idx}."
            )

        if not torch.isfinite(normalized_weights).all():
            raise ValueError("Sampling plan weights must be finite values.")
        if (normalized_weights < 0).any():
            raise ValueError("Sampling plan weights must be non-negative.")

        return normalized, normalized_weights

    def _apply_sample_filter(
        self,
        sample_filter: DatasetFilter[LatentFilterBatch],
    ) -> None:
        context = self._build_filter_context()
        sample_filter.prepare(context)
        plan = sample_filter.build_plan(context)
        validated_indices, validated_weights = self._validate_sampling_plan(plan)
        self.selected_source_indices = validated_indices
        self.selected_weights = validated_weights
        self.length = len(validated_indices)
        logger.info(
            "Applied sample filter to %s: selected %d / %d samples",
            self.zarr_path,
            self.length,
            self.source_length,
        )

    def _normalize_index(self, idx: int) -> int:
        idx_int = int(idx)
        if idx_int < 0:
            idx_int += self.length
        if idx_int < 0 or idx_int >= self.length:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.length}"
            )
        return idx_int

    def _map_filtered_to_source_index(self, idx: int) -> int:
        idx_int = self._normalize_index(idx)

        if self.selected_source_indices is None:
            return idx_int
        return self.selected_source_indices[idx_int]

    def _map_filtered_weight(self, idx: int) -> torch.Tensor:
        idx_int = self._normalize_index(idx)
        if self.selected_weights is None:
            return self._default_weight.clone()
        return self.selected_weights[idx_int]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        if self._mu_store is None:
            raise RuntimeError("Latent Zarr store not initialized.")

        source_idx = self._map_filtered_to_source_index(idx)
        weight = self._map_filtered_weight(idx)

        mu = torch.as_tensor(self._mu_store[source_idx], dtype=torch.float32)
        if self.use_logvar and self._logvar_store is not None:
            logvar = torch.as_tensor(
                self._logvar_store[source_idx],
                dtype=torch.float32,
            )
            # Add noise only to original (non-padded) region
            eps = torch.randn_like(mu)
            sample = mu + torch.exp(0.5 * logvar) * eps
            # Pad after sampling so padded regions are strictly zero
            return weight, self._pad_tensor(sample)
        # No logvar: just pad mu (padded regions will be zero)
        return weight, self._pad_tensor(mu)

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_d, pad_h, pad_w = self.pad_amounts
        if pad_d == pad_h == pad_w == 0:
            return tensor
        if tensor.ndim not in (4, 5):
            raise ValueError(
                "Expected tensor with 4D or 5D shape for padding, "
                f"got {tensor.ndim}D"
            )
        return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))
