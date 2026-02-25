"""Statistics loading and manipulation utilities."""

import json
from pathlib import Path
from typing import Dict

import torch


def load_stats(stats_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load statistics from a JSON file.

    Args:
        stats_path: Path to the JSON file.

    Returns:
        Dictionary with 'count', 'mean', 'std' tensors (and optionally 'mean_exp_logvar').
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


def reshape_stats_for_broadcast(
    stats: Dict[str, torch.Tensor],
    num_channels: int = 2,
    num_dims: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reshape mean and std for broadcasting to (B, C, D, H, W) tensors.

    Args:
        stats: Dictionary with 'mean' and 'std' tensors.
        num_channels: Number of channels in the data.
        num_dims: Total number of dimensions (default 5 for 3D volumes with batch).

    Returns:
        Tuple of (mean, std) tensors reshaped for broadcasting.
        Shape will be (1, C, 1, 1, 1) for num_dims=5.
    """
    # Build shape: (1, num_channels, 1, 1, ...) with (num_dims - 2) trailing 1s
    shape = (1, num_channels) + (1,) * (num_dims - 2)
    mean = stats["mean"].float().view(*shape)
    std = stats["std"].float().view(*shape)
    return mean, std


def get_effective_std(
    stats: Dict[str, torch.Tensor],
    use_logvar: bool,
) -> torch.Tensor:
    """
    Return std for normalization, optionally accounting for logvar sampling.

    When sampling z = mu + exp(0.5 * logvar) * eps, total variance is:
        Var[z] = Var[mu] + E[exp(logvar)]
    """
    std = stats["std"].float()
    if use_logvar:
        if "mean_exp_logvar" not in stats:
            raise ValueError(
                "mean_exp_logvar missing in stats; recompute latent stats with logvar."
            )
        variance = std.pow(2) + stats["mean_exp_logvar"].float()
        std = torch.sqrt(variance.clamp_min(0))
    return std


def get_global_std(
    stats: Dict[str, torch.Tensor],
    use_logvar: bool = False,
) -> torch.Tensor:
    """
    Compute global (scalar) standard deviation across all channels.
    
    This computes a single std value from per-channel stats by:
    1. Computing effective variance per channel (accounting for logvar if needed)
    2. Taking the mean of variances across channels
    3. Taking sqrt to get global std
    
    Args:
        stats: Dictionary with 'mean', 'std' tensors (and optionally 'mean_exp_logvar').
        use_logvar: If True, account for VAE sampling variance.
    
    Returns:
        Single scalar tensor with global std.
        
    Example:
        Given per-channel std [0.217, 0.631]:
        Global std = sqrt(mean([0.217², 0.631²])) = sqrt(0.2226) ≈ 0.472
    """
    # Get per-channel effective std
    per_channel_std = get_effective_std(stats, use_logvar)
    
    # Compute global std from mean of per-channel variances
    per_channel_variance = per_channel_std.pow(2)
    global_variance = per_channel_variance.mean()
    global_std = torch.sqrt(global_variance.clamp_min(0))
    
    return global_std
