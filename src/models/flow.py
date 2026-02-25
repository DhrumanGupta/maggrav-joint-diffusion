"""
Rectified Flow utilities for flow-based generative models.

Implements loss and sampling for rectified flow / flow matching.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .biflownet import BiFlowNet
from .unet import UNet3DDiffusion


def rectified_flow_loss(
    model: Union[UNet3DDiffusion, BiFlowNet],
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_per_sample: bool = False,
) -> torch.Tensor:
    """
    Rectified flow loss with uniform timestep sampling.

    x_t = (1 - t) * x + t * z0
    v = z0 - x
    loss = MSE(model(x_t, t), v)

    Args:
        model: UNet3DDiffusion model.
        x: Clean input tensor (B, C, D, H, W).

    Returns:
        Scalar loss tensor.
    """
    batch_size = x.shape[0]

    # Uniform sampling over [0, 1).
    t = torch.rand(batch_size, device=x.device)

    z0 = torch.randn_like(x)
    if mask is not None:
        z0 = z0 * mask

    t_bc = t[:, None, None, None, None]
    x_t = (1 - t_bc) * x + t_bc * z0
    if mask is not None:
        x_t = x_t * mask
    v = z0 - x
    v_pred = model(x_t, t)
    if mask is not None:
        v_pred = v_pred * mask

    diff_sq = (v_pred - v) ** 2
    if mask is not None:
        per_sample = (diff_sq * mask).sum(dim=(1, 2, 3, 4))
        denom = mask.sum() * x.shape[1]
        per_sample = per_sample / denom.clamp_min(1.0)
    else:
        per_sample = diff_sq.mean(dim=(1, 2, 3, 4))

    if return_per_sample:
        return per_sample
    return per_sample.mean()


@torch.no_grad()
def sample_rectified_flow(
    model: Union[UNet3DDiffusion, BiFlowNet],
    num_samples: int,
    latent_shape: Tuple[int, int, int, int],
    num_steps: int,
    device: torch.device,
    autocast_context=None,
    autocast_fn=None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Basic Euler sampler for rectified flow.

    Starts from x(t=1) ~ N(0, I) and integrates to t=0.

    Args:
        model: UNet3DDiffusion model.
        num_samples: Number of samples to generate.
        latent_shape: (C, D, H, W) shape of latents.
        num_steps: Number of integration steps.
        device: Torch device.
        autocast_context: Optional autocast context manager.
            Note: this should only be used for single-use contexts.
        autocast_fn: Optional callable returning an autocast context manager
            (e.g., accelerator.autocast). Preferred for iterative sampling loops.

    Returns:
        Generated samples tensor (B, C, D, H, W).
    """
    model.eval()
    channels, depth, height, width = latent_shape
    x = torch.randn(
        num_samples, channels, depth, height, width, device=device, dtype=torch.float32
    )
    if mask is not None:
        x = x * mask

    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t = t_vals[i]
        t_next = t_vals[i + 1]
        dt = t_next - t
        t_batch = torch.full((num_samples,), t, device=device)

        if autocast_fn is not None:
            with autocast_fn():
                v = model(x, t_batch)
        elif autocast_context is not None:
            with autocast_context:
                v = model(x, t_batch)
        else:
            v = model(x, t_batch)

        x = x + v * dt
        if mask is not None:
            x = x * mask

    model.train()
    return x
