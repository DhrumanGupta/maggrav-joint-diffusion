"""Padding mask utilities for handling variable-size volumes."""

from typing import Tuple

import torch


def create_padding_mask(
    padded_shape: Tuple[int, int, int, int],
    original_spatial: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a mask that is 1 in the valid (original) region and 0 in the padded region.

    Args:
        padded_shape: (C, D, H, W) - the padded tensor shape (channels + spatial).
        original_spatial: (D, H, W) - the original spatial dimensions before padding.
        device: Torch device for the mask tensor.

    Returns:
        Mask tensor of shape (1, 1, D, H, W) broadcastable to (B, C, D, H, W).
    """
    _, d_pad, h_pad, w_pad = padded_shape
    d_orig, h_orig, w_orig = original_spatial

    mask = torch.zeros(1, 1, d_pad, h_pad, w_pad, device=device)
    mask[:, :, :d_orig, :h_orig, :w_orig] = 1.0
    return mask
