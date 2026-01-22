"""
Shared utilities for computing and managing dataset statistics.

Provides streaming mean/std computation with DDP support, used for
normalizing data to zero mean and unit variance.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


def _stats_from_tensor(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-channel sum and sumsq for a batch tensor.

    Args:
        x: Input tensor of shape (B, C, D, H, W)
        mask: Optional mask tensor broadcastable to x. If provided, only
              masked (mask=1) elements are included in statistics.

    Returns:
        Tuple of (sum, sumsq, count) where:
            - sum: per-channel sum, shape (C,)
            - sumsq: per-channel sum of squares, shape (C,)
            - count: total number of elements per channel (scalar)
    """
    x = x.float()
    b, c, d, h, w = x.shape

    if mask is not None:
        # Expand mask to match x shape if needed
        if mask.dim() == 5 and mask.shape[1] == 1:
            # mask is (1, 1, D, H, W) or (B, 1, D, H, W)
            mask = mask.expand_as(x)
        elif mask.dim() != 5:
            raise ValueError(f"Mask must be 5D, got {mask.dim()}D")

        mask = mask.float()
        x_masked = x * mask

        # Reshape for reduction: (B, C, D*H*W)
        x_masked = x_masked.reshape(b, c, -1)
        mask_reshaped = mask.reshape(b, c, -1)

        # Count valid elements per channel (should be same across channels)
        count = mask_reshaped[:, 0, :].sum()  # Use first channel's mask
        sum_ = x_masked.sum(dim=(0, 2))  # (C,)
        sumsq = (x_masked**2).sum(dim=(0, 2))
    else:
        x = x.reshape(b, c, -1)
        count = torch.tensor(b * x.shape[2], dtype=torch.float32, device=x.device)
        sum_ = x.sum(dim=(0, 2))
        sumsq = (x * x).sum(dim=(0, 2))

    return sum_, sumsq, count


def _variance_from_logvar(
    logvar: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel sum of exp(logvar) for variance correction.

    When sampling z = mu + exp(0.5 * logvar) * eps, the total variance is:
        Var[z] = Var[mu] + E[exp(logvar)]

    This function computes the sum of exp(logvar) to later compute E[exp(logvar)].

    Args:
        logvar: Log-variance tensor of shape (B, C, D, H, W)
        mask: Optional mask tensor broadcastable to logvar

    Returns:
        Tuple of (sum_exp_logvar, count) where:
            - sum_exp_logvar: per-channel sum of exp(logvar), shape (C,)
            - count: total number of elements per channel (scalar)
    """
    logvar = logvar.float()
    b, c, d, h, w = logvar.shape
    var = torch.exp(logvar)

    if mask is not None:
        if mask.dim() == 5 and mask.shape[1] == 1:
            mask = mask.expand_as(var)
        elif mask.dim() != 5:
            raise ValueError(f"Mask must be 5D, got {mask.dim()}D")

        mask = mask.float()
        var_masked = var * mask
        var_masked = var_masked.reshape(b, c, -1)
        mask_reshaped = mask.reshape(b, c, -1)

        count = mask_reshaped[:, 0, :].sum()
        sum_var = var_masked.sum(dim=(0, 2))
    else:
        var = var.reshape(b, c, -1)
        count = torch.tensor(b * var.shape[2], dtype=torch.float32, device=var.device)
        sum_var = var.sum(dim=(0, 2))

    return sum_var, count


def compute_streaming_stats_ddp(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    accelerator: Accelerator,
    num_channels: int,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute streaming mean/std statistics across a dataset with DDP support.

    Uses Welford-style online algorithm to compute exact statistics without
    loading entire dataset into memory. All-reduces across processes for
    exact global statistics.

    Args:
        dataset: Dataset to compute statistics over
        batch_size: Batch size for iteration
        num_workers: Number of data loader workers
        accelerator: Accelerate instance for DDP coordination
        num_channels: Number of channels in the data
        mask: Optional mask tensor for excluding padded regions

    Returns:
        Dictionary with 'count', 'mean', 'std' tensors
    """
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

    loader = DataLoader(dataset, **loader_kwargs)
    if sampler is not None:
        sampler.set_epoch(0)

    device = accelerator.device
    count = torch.zeros(1, dtype=torch.float32, device=device)
    sum_ = torch.zeros(num_channels, dtype=torch.float32, device=device)
    sumsq = torch.zeros(num_channels, dtype=torch.float32, device=device)

    # Move mask to device if provided
    if mask is not None:
        mask = mask.to(device)

    progress = tqdm(
        loader,
        desc="Computing mean/std",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        batch_sum, batch_sumsq, batch_count = _stats_from_tensor(batch, mask)
        sum_ += batch_sum
        sumsq += batch_sumsq
        count += batch_count

    # All-reduce across processes for exact global sums
    sum_ = accelerator.reduce(sum_, reduction="sum")
    sumsq = accelerator.reduce(sumsq, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")

    mean = sum_ / count
    var = sumsq / count - mean.pow(2)
    std = torch.sqrt(var.clamp_min(0))

    return {
        "count": count.cpu(),
        "mean": mean.cpu(),
        "std": std.cpu(),
    }


def compute_latent_stats_ddp(
    zarr_path: str,
    batch_size: int,
    num_workers: int,
    accelerator: Accelerator,
    num_channels: int,
    pad_to: int = 32,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute streaming mean/std statistics for VAE latents with DDP support.

    This function correctly accounts for the variance when sampling from
    the VAE posterior: z = mu + exp(0.5 * logvar) * eps

    The total variance is: Var[z] = Var[mu] + E[exp(logvar)]

    Args:
        zarr_path: Path to the latent zarr store
        batch_size: Batch size for iteration
        num_workers: Number of data loader workers
        accelerator: Accelerate instance for DDP coordination
        num_channels: Number of channels in the latents
        pad_to: Padding size for spatial dimensions
        mask: Optional mask tensor for excluding padded regions

    Returns:
        Dictionary with 'count', 'mean', 'std', 'mean_exp_logvar' tensors
    """
    import torch.nn.functional as F
    import zarr
    from torch.utils.data import Dataset as TorchDataset

    # Check if logvar exists
    group = zarr.open_group(zarr_path, mode="r")
    has_logvar = "latent_logvar" in group

    # Simple dataset that returns (mu, logvar) or just mu
    class LatentMuLogvarDataset(TorchDataset):
        def __init__(self, zarr_path: str, pad_to: int, return_logvar: bool):
            self.zarr_path = zarr_path
            self.pad_to = pad_to
            self.return_logvar = return_logvar
            self._group = None
            self._mu_store = None
            self._logvar_store = None

            group = zarr.open_group(zarr_path, mode="r")
            mu_store = group["latent_mu"]
            self.length = mu_store.shape[0]
            self.original_spatial = mu_store.shape[2:]
            self.padded_spatial = tuple(
                max(dim, pad_to) for dim in self.original_spatial
            )
            self.pad_amounts = tuple(
                padded - orig
                for padded, orig in zip(self.padded_spatial, self.original_spatial)
            )

        def _ensure_open(self):
            if self._group is None:
                self._group = zarr.open_group(self.zarr_path, mode="r")
                self._mu_store = self._group["latent_mu"]
                if self.return_logvar and "latent_logvar" in self._group:
                    self._logvar_store = self._group["latent_logvar"]

        def __len__(self):
            return self.length

        def _pad_tensor(self, tensor):
            pad_d, pad_h, pad_w = self.pad_amounts
            if pad_d == pad_h == pad_w == 0:
                return tensor
            return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))

        def __getitem__(self, idx):
            self._ensure_open()
            mu = torch.as_tensor(self._mu_store[idx], dtype=torch.float32)
            mu = self._pad_tensor(mu)
            if self.return_logvar and self._logvar_store is not None:
                logvar = torch.as_tensor(self._logvar_store[idx], dtype=torch.float32)
                logvar = self._pad_tensor(logvar)
                return mu, logvar
            return mu

    if accelerator.num_processes > 1:
        # Create dataset for length calculation
        temp_dataset = LatentMuLogvarDataset(zarr_path, pad_to, return_logvar=False)
        sampler = DistributedSampler(
            temp_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    # Create dataset that returns both mu and logvar if available
    dataset = LatentMuLogvarDataset(zarr_path, pad_to, return_logvar=has_logvar)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

    loader = DataLoader(dataset, **loader_kwargs)
    if sampler is not None:
        sampler.set_epoch(0)

    device = accelerator.device
    count = torch.zeros(1, dtype=torch.float32, device=device)
    sum_ = torch.zeros(num_channels, dtype=torch.float32, device=device)
    sumsq = torch.zeros(num_channels, dtype=torch.float32, device=device)
    sum_exp_logvar = torch.zeros(num_channels, dtype=torch.float32, device=device)

    if mask is not None:
        mask = mask.to(device)

    progress = tqdm(
        loader,
        desc="Computing latent stats",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress:
        if has_logvar:
            mu, logvar = batch
            mu = mu.to(device, non_blocking=True)
            logvar = logvar.to(device, non_blocking=True)

            # Compute mu stats
            batch_sum, batch_sumsq, batch_count = _stats_from_tensor(mu, mask)
            sum_ += batch_sum
            sumsq += batch_sumsq
            count += batch_count

            # Compute sum of exp(logvar) for variance correction
            batch_sum_var, _ = _variance_from_logvar(logvar, mask)
            sum_exp_logvar += batch_sum_var
        else:
            mu = batch.to(device, non_blocking=True)
            batch_sum, batch_sumsq, batch_count = _stats_from_tensor(mu, mask)
            sum_ += batch_sum
            sumsq += batch_sumsq
            count += batch_count

    # All-reduce across processes
    sum_ = accelerator.reduce(sum_, reduction="sum")
    sumsq = accelerator.reduce(sumsq, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    sum_exp_logvar = accelerator.reduce(sum_exp_logvar, reduction="sum")

    mean = sum_ / count
    var_mu = sumsq / count - mean.pow(2)

    # Total variance when sampling: Var[z] = Var[mu] + E[exp(logvar)]
    mean_exp_logvar = sum_exp_logvar / count if has_logvar else torch.zeros_like(mean)
    total_var = var_mu + mean_exp_logvar
    std = torch.sqrt(total_var.clamp_min(0))

    return {
        "count": count.cpu(),
        "mean": mean.cpu(),
        "std": std.cpu(),
        "mean_exp_logvar": mean_exp_logvar.cpu(),  # Store for reference
    }


def load_stats(stats_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load statistics from a JSON file.

    Args:
        stats_path: Path to the JSON file

    Returns:
        Dictionary with 'count', 'mean', 'std' tensors (and optionally 'mean_exp_logvar')
    """
    with stats_path.open("r") as f:
        payload = json.load(f)
    result = {
        "count": torch.tensor(payload["count"], dtype=torch.float64),
        "mean": torch.tensor(payload["mean"], dtype=torch.float64),
        "std": torch.tensor(payload["std"], dtype=torch.float64),
    }
    # Optionally load mean_exp_logvar if present (for latent stats)
    if "mean_exp_logvar" in payload:
        result["mean_exp_logvar"] = torch.tensor(
            payload["mean_exp_logvar"], dtype=torch.float64
        )
    return result


def save_stats(stats_path: Path, stats: Dict[str, torch.Tensor]) -> None:
    """
    Save statistics to a JSON file.

    Args:
        stats_path: Path to save the JSON file
        stats: Dictionary with 'count', 'mean', 'std' tensors (and optionally 'mean_exp_logvar')
    """
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": stats["count"].tolist(),
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
    }
    # Optionally save mean_exp_logvar if present (for latent stats)
    if "mean_exp_logvar" in stats:
        payload["mean_exp_logvar"] = stats["mean_exp_logvar"].tolist()
    with stats_path.open("w") as f:
        json.dump(payload, f, indent=2)
