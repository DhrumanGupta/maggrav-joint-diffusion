"""
Rectified Flow utilities for flow-based generative models.

Implements loss and sampling for rectified flow / flow matching.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .unet import UNet3DDiffusion


def rectified_flow_loss(
    model: UNet3DDiffusion,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """
    Rectified flow loss with logit-normal timestep sampling.

    Logit-normal biases sampling toward middle timesteps (t ~ 0.5),
    which are harder to predict and more informative for training.
    This follows the approach used in Stable Diffusion 3.

    x_t = (1 - t) * x + t * z0
    v = z0 - x
    loss = MSE(model(x_t, t), v)

    If mask is provided, noise is only added to the valid region and
    loss is only computed on the valid region.

    Args:
        model: UNet3DDiffusion model.
        x: Clean input tensor (B, C, D, H, W).
        mask: Optional mask for valid regions (1=valid, 0=padded).
        logit_mean: Mean of logit-normal distribution.
        logit_std: Std of logit-normal distribution.

    Returns:
        Scalar loss tensor.
    """
    batch_size = x.shape[0]

    # Logit-normal sampling: sample from normal, then apply sigmoid
    u = torch.randn(batch_size, device=x.device) * logit_std + logit_mean
    t = torch.sigmoid(u)  # t in (0, 1), concentrated around 0.5

    z0 = torch.randn_like(x)

    # If mask is provided, zero out noise in padded regions
    if mask is not None:
        z0 = z0 * mask

    t_bc = t[:, None, None, None, None]
    x_t = (1 - t_bc) * x + t_bc * z0
    v = z0 - x
    v_pred = model(x_t, t)

    # Compute loss only on valid (non-padded) regions
    if mask is not None:
        # Masked MSE loss: average over valid elements only
        diff_sq = (v_pred - v) ** 2
        masked_diff_sq = diff_sq * mask
        loss = masked_diff_sq.sum() / mask.sum() / x.shape[1]  # normalize by channels
    else:
        loss = F.mse_loss(v_pred, v)

    return loss


@torch.no_grad()
def sample_rectified_flow(
    model: UNet3DDiffusion,
    num_samples: int,
    latent_shape: Tuple[int, int, int, int],
    num_steps: int,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
    autocast_context=None,
) -> torch.Tensor:
    """
    Basic Euler sampler for rectified flow.

    Starts from x(t=1) ~ N(0, I) and integrates to t=0.

    If mask is provided, noise is only in the valid region and
    padded regions stay at zero throughout.

    Args:
        model: UNet3DDiffusion model.
        num_samples: Number of samples to generate.
        latent_shape: (C, D, H, W) shape of latents.
        num_steps: Number of integration steps.
        device: Torch device.
        mask: Optional mask for valid regions.
        autocast_context: Optional autocast context manager (e.g., accelerator.autocast()).

    Returns:
        Generated samples tensor (B, C, D, H, W).
    """
    model.eval()
    channels, depth, height, width = latent_shape
    x = torch.randn(
        num_samples, channels, depth, height, width, device=device, dtype=torch.float32
    )

    # Apply mask to initial noise if provided
    if mask is not None:
        x = x * mask

    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t = t_vals[i]
        t_next = t_vals[i + 1]
        dt = t_next - t
        t_batch = torch.full((num_samples,), t, device=device)

        if autocast_context is not None:
            with autocast_context:
                v = model(x, t_batch)
        else:
            v = model(x, t_batch)

        x = x + v * dt

        # Keep padded regions at zero after each step
        if mask is not None:
            x = x * mask

    model.train()
    return x
