"""Checkpoint loading and state dict utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    config_classes: Optional[List[Type]] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint with safe deserialization of config dataclasses.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to map tensors to.
        config_classes: List of config dataclass types to allow during loading.
                       If None, falls back to weights_only=False.

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config_classes:
        try:
            from torch.serialization import safe_globals

            with safe_globals(config_classes):
                return torch.load(checkpoint_path, map_location=device)
        except Exception:
            pass

    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def clean_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Remove '_orig_mod.' prefix from state dict keys (added by torch.compile).

    Args:
        state_dict: Model state dictionary potentially with '_orig_mod.' prefixes.

    Returns:
        Cleaned state dictionary.
    """
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return {
            key.replace("_orig_mod.", "", 1): val for key, val in state_dict.items()
        }
    return state_dict
