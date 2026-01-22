"""Common dataset classes for density/susceptibility and latent data."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

from ..data.zarr_dataset import NoddyverseZarrDataset

logger = logging.getLogger(__name__)


class JointDensitySuscDataset(Dataset):
    """
    Dataset that returns stacked density and susceptibility tensors.

    Susceptibility is always log10-transformed: log10(susc + 1e-6).
    Density is returned as-is.

    Args:
        zarr_path: Path to the Noddyverse Zarr store.
        return_index: If True, returns (index, tensor) tuple instead of just tensor.
    """

    def __init__(self, zarr_path: str, return_index: bool = False) -> None:
        self.base = NoddyverseZarrDataset(
            zarr_path,
            fields=("rock_types",),
            include_metadata=False,
            return_tensors=True,
        )
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[int, torch.Tensor]]:
        sample = self.base[idx]
        density = sample["density"]
        susceptibility = sample["susceptibility"]
        # Always apply log10 transformation to susceptibility
        susceptibility = torch.log10(susceptibility + 1e-6)
        stacked = torch.stack([density, susceptibility], dim=0)

        if self.return_index:
            return idx, stacked
        return stacked


class LatentZarrDataset(Dataset):
    """
    Dataset that samples latents from a Zarr store on each access.

    Supports optional padding to a minimum spatial size and stochastic
    sampling from the VAE posterior when logvar is available.

    Args:
        zarr_path: Path to the latent Zarr store.
        pad_to: Minimum spatial dimension size (pads with zeros if smaller).
        use_logvar: If True and logvar is available, sample z = mu + exp(0.5*logvar)*eps.
    """

    def __init__(self, zarr_path: str, pad_to: int = 32, use_logvar: bool = True):
        self.zarr_path = zarr_path
        self.pad_to = max(pad_to, 1)

        group = zarr.open_group(zarr_path, mode="r")
        if "latent_mu" not in group:
            raise ValueError(f"Zarr store missing 'latent_mu': {zarr_path}")

        mu_store = group["latent_mu"]
        self.length = mu_store.shape[0]
        self.channels = mu_store.shape[1]
        self.original_spatial: Tuple[int, int, int] = mu_store.shape[2:]
        self.padded_spatial = self._compute_padded_spatial(self.original_spatial)
        self.pad_amounts = tuple(
            padded - orig
            for padded, orig in zip(self.padded_spatial, self.original_spatial)
        )
        self.latent_shape = (self.channels,) + self.padded_spatial
        self.has_logvar = "latent_logvar" in group
        self.use_logvar = use_logvar and self.has_logvar

        if use_logvar and not self.has_logvar:
            logger.warning(
                "Latent logvar not found in %s; sampling deterministically from mu.",
                zarr_path,
            )

        self._group: Optional[zarr.Group] = None
        self._mu_store: Optional[zarr.Array] = None
        self._logvar_store: Optional[zarr.Array] = None

    def _compute_padded_spatial(
        self, spatial: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        pad_to = self.pad_to
        return tuple(max(dim, pad_to) for dim in spatial)

    def _ensure_open(self) -> None:
        if self._group is None:
            group = zarr.open_group(self.zarr_path, mode="r")
            self._group = group
            self._mu_store = group["latent_mu"]
            self._logvar_store = (
                group["latent_logvar"] if "latent_logvar" in group else None
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_open()
        if self._mu_store is None:
            raise RuntimeError("Latent Zarr store not initialized.")

        mu = torch.as_tensor(self._mu_store[idx], dtype=torch.float32)
        if self.use_logvar and self._logvar_store is not None:
            logvar = torch.as_tensor(self._logvar_store[idx], dtype=torch.float32)
            # Add noise only to original (non-padded) region
            eps = torch.randn_like(mu)
            sample = mu + torch.exp(0.5 * logvar) * eps
            # Pad after sampling so padded regions are strictly zero
            return self._pad_tensor(sample)
        # No logvar: just pad mu (padded regions will be zero)
        return self._pad_tensor(mu)

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_d, pad_h, pad_w = self.pad_amounts
        if pad_d == pad_h == pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))
