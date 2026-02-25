"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models) utilities.

Implements preconditioning, loss, and sampling from Karras et al., 2022.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .unet import UNet3DDiffusion
from .dit import DiT3D


class EDMPrecond(nn.Module):
    """
    EDM preconditioning wrapper for the UNet or DiT.

    Implements the preconditioning from "Elucidating the Design Space of
    Diffusion-Based Generative Models" (Karras et al., 2022).

    The raw network F_θ is wrapped with input/output scaling:
        D_θ(x; σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x; c_noise(σ))

    where:
        c_skip(σ) = σ_data² / (σ² + σ_data²)
        c_out(σ) = σ * σ_data / sqrt(σ² + σ_data²)
        c_in(σ) = 1 / sqrt(σ² + σ_data²)
        c_noise(σ) = ln(σ) / 4

    Args:
        model: The underlying UNet3DDiffusion model.
        sigma_data: Data standard deviation (default 0.5 for unnormalized, 1.0 for normalized).
    """

    def __init__(self, model: Union[UNet3DDiffusion, DiT3D], sigma_data: float = 0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Apply EDM preconditioning and run the model.

        Args:
            x: Noisy input tensor (B, C, D, H, W).
            sigma: Noise levels (B,) - can be any positive value.

        Returns:
            Denoised estimate (B, C, D, H, W).
        """
        sigma = sigma.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1) for broadcasting
        sigma_data = self.sigma_data

        # Preconditioning coefficients
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
        c_noise = sigma.view(-1).log() / 4  # (B,) for time embedding

        # Apply preconditioning
        scaled_input = c_in * x
        F_out = self.model(scaled_input, c_noise)
        denoised = c_skip * x + c_out * F_out

        return denoised


def edm_loss(
    model: EDMPrecond,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_data: float = 0.5,
    return_per_sample: bool = False,
) -> torch.Tensor:
    """
    EDM training loss with log-normal sigma sampling.

    From "Elucidating the Design Space of Diffusion-Based Generative Models":
    - Sample sigma from log-normal: ln(σ) ~ N(P_mean, P_std²)
    - Create noisy samples: x_noisy = x + σ * ε
    - Loss weight: λ(σ) = (σ² + σ_data²) / (σ * σ_data)²
    - Loss = λ(σ) * ||D_θ(x_noisy; σ) - x||²

    If mask is provided, noise is only added to the valid region and
    loss is only computed on the valid region.

    Args:
        model: EDMPrecond-wrapped model.
        x: Clean input tensor (B, C, D, H, W).
        mask: Optional mask for valid regions (1=valid, 0=padded).
        P_mean: Mean of log-normal distribution for sigma.
        P_std: Std of log-normal distribution for sigma.
        sigma_data: Data standard deviation.

    Returns:
        Scalar loss tensor by default. If ``return_per_sample`` is True,
        returns a tensor of shape (B,) with per-sample losses.
    """
    batch_size = x.shape[0]
    device = x.device

    # Sample sigma from log-normal distribution
    # ln(σ) ~ N(P_mean, P_std²) => σ = exp(P_mean + P_std * z), z ~ N(0,1)
    ln_sigma = torch.randn(batch_size, device=device) * P_std + P_mean
    sigma = ln_sigma.exp()  # (B,)

    # Sample noise
    noise = torch.randn_like(x)

    # If mask is provided, zero out noise in padded regions
    if mask is not None:
        noise = noise * mask

    # Create noisy samples: x_noisy = x + σ * ε
    sigma_bc = sigma.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
    x_noisy = x + sigma_bc * noise

    # Get denoised prediction
    denoised = model(x_noisy, sigma)

    # Loss weight: λ(σ) = (σ² + σ_data²) / (σ * σ_data)²
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    weight = weight.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)

    # Compute loss only on valid (non-padded) regions
    if mask is not None:
        # Average per sample over valid voxels/channels, then average batch.
        diff_sq = (denoised - x) ** 2
        weighted_diff_sq = weight * diff_sq * mask
        valid_per_sample = mask.sum() * x.shape[1]
        per_sample_loss = weighted_diff_sq.sum(dim=(1, 2, 3, 4)) / valid_per_sample
    else:
        diff_sq = (denoised - x) ** 2
        per_sample_loss = (weight * diff_sq).mean(dim=(1, 2, 3, 4))

    if return_per_sample:
        return per_sample_loss
    return per_sample_loss.mean()


def get_karras_sigmas(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate Karras et al. noise schedule for EDM sampling.

    σ_i = (σ_max^(1/ρ) + i/(n-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

    Args:
        num_steps: Number of sampling steps.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        rho: Schedule curvature parameter.
        device: Torch device.

    Returns:
        Sigma values from sigma_max to sigma_min (plus final 0).
    """
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # Append 0 at the end for the final step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
    return sigmas


@torch.no_grad()
def sample_edm_heun(
    model: EDMPrecond,
    num_samples: int,
    latent_shape: Tuple[int, int, int, int],
    num_steps: int,
    device: torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    mask: Optional[torch.Tensor] = None,
    autocast_context=None,
    autocast_fn=None,
) -> torch.Tensor:
    """
    Heun's 2nd-order deterministic sampler for EDM.

    Starts from x ~ N(0, σ_max²) and integrates to σ_min using the
    probability flow ODE with Heun's method (predictor-corrector).

    Args:
        model: EDMPrecond-wrapped model.
        num_samples: Number of samples to generate.
        latent_shape: (C, D, H, W) shape of latents.
        num_steps: Number of sampling steps.
        device: Torch device.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        rho: Schedule curvature parameter.
        mask: Optional mask for valid regions.
        autocast_context: Optional autocast context manager.
            Note: this should only be used for single-use contexts.
        autocast_fn: Optional callable returning an autocast context manager
            (e.g., accelerator.autocast). Preferred for iterative sampling loops.

    Returns:
        Generated samples tensor (B, C, D, H, W).
    """
    model.eval()
    channels, depth, height, width = latent_shape

    # Get sigma schedule
    sigmas = get_karras_sigmas(num_steps, sigma_min, sigma_max, rho, device)

    # Initialize from N(0, σ_max²)
    x = torch.randn(num_samples, channels, depth, height, width, device=device)
    x = x * sigmas[0]  # Scale by σ_max

    # Apply mask to initial noise if provided
    if mask is not None:
        x = x * mask

    for i in range(len(sigmas) - 1):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Get denoised estimate at current sigma
        sigma_batch = sigma_cur.expand(num_samples)
        if autocast_fn is not None:
            with autocast_fn():
                denoised = model(x, sigma_batch)
        elif autocast_context is not None:
            with autocast_context:
                denoised = model(x, sigma_batch)
        else:
            denoised = model(x, sigma_batch)

        # Derivative: d = (x - D(x, σ)) / σ
        d_cur = (x - denoised) / sigma_cur

        # Euler step (predictor)
        x_next = x + (sigma_next - sigma_cur) * d_cur

        # Heun correction (if not at final step)
        if sigma_next > 0:
            sigma_batch_next = sigma_next.expand(num_samples)
            if autocast_fn is not None:
                with autocast_fn():
                    denoised_next = model(x_next, sigma_batch_next)
            elif autocast_context is not None:
                with autocast_context:
                    denoised_next = model(x_next, sigma_batch_next)
            else:
                denoised_next = model(x_next, sigma_batch_next)
            d_next = (x_next - denoised_next) / sigma_next
            # Average the derivatives
            x_next = x + (sigma_next - sigma_cur) * (d_cur + d_next) / 2

        x = x_next

        # Keep padded regions at zero after each step
        if mask is not None:
            x = x * mask

    model.train()
    return x
