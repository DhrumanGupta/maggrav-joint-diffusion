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
