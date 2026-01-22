"""
Sample unconditional volumes from an EDM diffusion checkpoint and decode with a VAE.

Usage:
    python src/sample_unconditional_diffusion.py \
        --checkpoint outputs/edm_150m/edm_checkpoint_step_1000000.pt \
        --vae_checkpoint outputs_vae/vae_checkpoint_step_1000000.pt \
        --num_samples 4 \
        --vae_stats_path vae_stats.json

The EDM checkpoint should contain latent_mean and latent_std for denormalizing
the generated latents back to the original latent space before VAE decoding.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed

from models.unet import UNet3DConfig, UNet3DDiffusion
from models.vae import VAE3D, VAE3DConfig

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# -------------------------
# EDM Preconditioning
# -------------------------


class EDMPrecond(nn.Module):
    """
    EDM preconditioning wrapper for the UNet.

    Implements the preconditioning from "Elucidating the Design Space of
    Diffusion-Based Generative Models" (Karras et al., 2022).

    The raw network F_θ is wrapped with input/output scaling:
        D_θ(x; σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x; c_noise(σ))

    where:
        c_skip(σ) = σ_data² / (σ² + σ_data²)
        c_out(σ) = σ * σ_data / sqrt(σ² + σ_data²)
        c_in(σ) = 1 / sqrt(σ² + σ_data²)
        c_noise(σ) = ln(σ) / 4
    """

    def __init__(self, model: UNet3DDiffusion, sigma_data: float = 0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy input tensor (B, C, D, H, W)
            sigma: Noise levels (B,) - can be any positive value

        Returns:
            Denoised estimate (B, C, D, H, W)
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


# -------------------------
# Helper Functions
# -------------------------


def _load_checkpoint(checkpoint_path: Path, device: torch.device, config_cls) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        from torch.serialization import safe_globals

        with safe_globals([config_cls]):
            return torch.load(checkpoint_path, map_location=device)
    except Exception:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_stats(stats_path: Path) -> Dict[str, torch.Tensor]:
    with stats_path.open("r") as f:
        payload = json.load(f)
    return {
        "count": torch.tensor(payload["count"], dtype=torch.float64),
        "mean": torch.tensor(payload["mean"], dtype=torch.float64),
        "std": torch.tensor(payload["std"], dtype=torch.float64),
    }


def create_padding_mask(
    padded_shape: Tuple[int, int, int, int],
    original_spatial: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a mask that is 1 in the valid (original) region and 0 in the padded region.
    """
    _, d_pad, h_pad, w_pad = padded_shape
    d_orig, h_orig, w_orig = original_spatial

    mask = torch.zeros(1, 1, d_pad, h_pad, w_pad, device=device)
    mask[:, :, :d_orig, :h_orig, :w_orig] = 1.0
    return mask


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return {
            key.replace("_orig_mod.", "", 1): val for key, val in state_dict.items()
        }
    return state_dict


# -------------------------
# EDM Sampling
# -------------------------


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

    Returns sigma values from sigma_max to sigma_min (plus final 0).
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
    accelerator: Accelerator,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Heun's 2nd-order deterministic sampler for EDM.

    Starts from x ~ N(0, σ_max²) and integrates to σ_min using the
    probability flow ODE with Heun's method (predictor-corrector).

    If mask is provided, noise is only in the valid region and
    padded regions stay at zero throughout.
    """
    model.eval()
    channels, depth, height, width = latent_shape
    device = accelerator.device

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
        with accelerator.autocast():
            denoised = model(x, sigma_batch)

        # Derivative: d = (x - D(x, σ)) / σ
        d_cur = (x - denoised) / sigma_cur

        # Euler step (predictor)
        x_next = x + (sigma_next - sigma_cur) * d_cur

        # Heun correction (if not at final step)
        if sigma_next > 0:
            sigma_batch_next = sigma_next.expand(num_samples)
            with accelerator.autocast():
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


# -------------------------
# Plotting
# -------------------------


def _plot_sample_slices(
    volume: np.ndarray,
    output_path: Path,
    z_indices: list[int],
    channel_names: Tuple[str, str],
) -> None:
    num_channels = volume.shape[0]
    num_cols = len(z_indices)
    fig, axes = plt.subplots(
        num_channels,
        num_cols,
        figsize=(1.8 * num_cols, 2.2 * num_channels),
        squeeze=False,
    )

    for c in range(num_channels):
        channel = np.nan_to_num(volume[c], nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(channel.min())
        vmax = float(channel.max())
        for col, z in enumerate(z_indices):
            ax = axes[c, col]
            ax.imshow(channel[z], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{channel_names[c]} z={z}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _crop_latents(
    latents: torch.Tensor, target_shape: Tuple[int, int, int, int]
) -> torch.Tensor:
    channels, depth, height, width = target_shape
    return latents[:, :channels, :depth, :height, :width]


# -------------------------
# Arguments
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample unconditional volumes from an EDM diffusion checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument(
        "--vae_stats_path",
        type=str,
        default="vae_stats.json",
        help="Path to VAE input data stats (density/susceptibility normalization)",
    )
    parser.add_argument(
        "--latent_stats_path",
        type=str,
        default=None,
        help="Path to latent stats JSON (optional, loaded from checkpoint if available)",
    )
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    # EDM-specific sampling parameters
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum noise level for sampling",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum noise level for sampling",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="Karras schedule curvature parameter",
    )
    parser.add_argument(
        "--decode_device",
        type=str,
        choices=("cuda", "cpu"),
        default=None,
        help="Device for VAE decoding (defaults to accelerator device if available).",
    )
    parser.add_argument("--output_dir", type=str, default="outputs_edm_samples")
    return parser.parse_args()


def _sample_range(
    num_samples: int, num_processes: int, process_index: int
) -> tuple[int, int]:
    if num_processes < 1:
        return 0, num_samples
    per_rank = int(math.ceil(num_samples / num_processes))
    start = process_index * per_rank
    end = min(start + per_rank, num_samples)
    return start, end


# -------------------------
# Main
# -------------------------


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed + accelerator.process_index)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = accelerator.device
    edm_ckpt = _load_checkpoint(Path(args.checkpoint), device, UNet3DConfig)
    edm_cfg = edm_ckpt["config"]

    # Get sigma_data from checkpoint (default 1.0 for normalized latents)
    sigma_data = edm_ckpt.get("sigma_data", 1.0)
    if accelerator.is_main_process:
        logger.info("Using sigma_data=%.4f from checkpoint", sigma_data)

    # Create EDMPrecond-wrapped model
    raw_model = UNet3DDiffusion(edm_cfg)
    edm_model = EDMPrecond(raw_model, sigma_data=sigma_data)
    edm_state = _clean_state_dict(edm_ckpt["model_state_dict"])
    edm_model.load_state_dict(edm_state)
    edm_model.to(device)

    # Load VAE stats (for denormalizing decoded outputs)
    vae_stats_path = Path(args.vae_stats_path)
    if not vae_stats_path.exists():
        raise FileNotFoundError(f"VAE stats file not found: {vae_stats_path}")
    vae_stats = load_stats(vae_stats_path)
    vae_mean_cpu = vae_stats["mean"].float().view(1, 2, 1, 1, 1)
    vae_std_cpu = vae_stats["std"].float().view(1, 2, 1, 1, 1)

    # Load latent stats (for denormalizing sampled latents)
    # Priority: checkpoint > explicit file > None (no denormalization for old checkpoints)
    latent_mean = None
    latent_std = None
    if "latent_mean" in edm_ckpt and "latent_std" in edm_ckpt:
        # Load from checkpoint (preferred)
        latent_mean = edm_ckpt["latent_mean"].float().to(device)
        latent_std = edm_ckpt["latent_std"].float().to(device)
        if accelerator.is_main_process:
            logger.info("Loaded latent stats from checkpoint")
    elif args.latent_stats_path is not None:
        # Load from explicit file
        latent_stats_path = Path(args.latent_stats_path)
        if not latent_stats_path.exists():
            raise FileNotFoundError(f"Latent stats file not found: {latent_stats_path}")
        latent_stats = load_stats(latent_stats_path)
        num_latent_channels = edm_cfg.in_channels
        latent_mean = (
            latent_stats["mean"]
            .float()
            .view(1, num_latent_channels, 1, 1, 1)
            .to(device)
        )
        latent_std = (
            latent_stats["std"].float().view(1, num_latent_channels, 1, 1, 1).to(device)
        )
        if accelerator.is_main_process:
            logger.info("Loaded latent stats from %s", latent_stats_path)
    else:
        if accelerator.is_main_process:
            logger.warning(
                "No latent stats found in checkpoint or provided via --latent_stats_path. "
                "Assuming latents are not normalized (old checkpoint)."
            )

    # Get latent shape - prefer from checkpoint, fallback to config
    if "latent_shape" in edm_ckpt:
        latent_shape = tuple(edm_ckpt["latent_shape"])
    else:
        latent_shape = (
            edm_cfg.in_channels,
            edm_cfg.input_spatial,
            edm_cfg.input_spatial,
            edm_cfg.input_spatial,
        )

    # Get original spatial dimensions for cropping and mask
    if "original_spatial" in edm_ckpt:
        original_spatial = tuple(edm_ckpt["original_spatial"])
    else:
        # Fallback for older checkpoints - assume no padding
        original_spatial = latent_shape[1:]

    # Create padding mask if there is padding
    padding_mask = None
    if original_spatial != latent_shape[1:]:
        padding_mask = create_padding_mask(
            padded_shape=latent_shape,
            original_spatial=original_spatial,
            device=device,
        )
        if accelerator.is_main_process:
            logger.info(
                "Padding mask created: original %s -> padded %s",
                original_spatial,
                latent_shape[1:],
            )

    z_depth = original_spatial[0]  # Use original depth for plotting
    z_max = min(190, z_depth - 1)
    z_indices = list(range(0, z_max + 1, 10))
    if z_indices[-1] != z_max:
        z_indices.append(z_max)
    if accelerator.is_main_process:
        logger.info("Depth=%d, plotting slices: %s", z_depth, z_indices)

    channel_names = ("grav", "mag")
    total = args.num_samples
    batch_size = max(1, args.batch_size)
    start_idx, end_idx = _sample_range(
        total, accelerator.num_processes, accelerator.process_index
    )
    has_work = start_idx < end_idx

    generated = start_idx

    if accelerator.is_main_process:
        logger.info(
            "Sampling latents with EDM model (sigma_min=%.4f, sigma_max=%.1f, rho=%.1f, steps=%d).",
            args.sigma_min,
            args.sigma_max,
            args.rho,
            args.num_steps,
        )
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    # Target shape for cropping (channels, D, H, W)
    crop_target = (latent_shape[0],) + original_spatial

    if has_work:
        while generated < end_idx:
            batch = min(batch_size, end_idx - generated)
            with torch.no_grad():
                latents = sample_edm_heun(
                    edm_model,
                    num_samples=batch,
                    latent_shape=latent_shape,
                    num_steps=args.num_steps,
                    accelerator=accelerator,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    rho=args.rho,
                    mask=padding_mask,
                )
            # Denormalize latents back to original latent space
            if latent_mean is not None and latent_std is not None:
                latents = latents * latent_std + latent_mean
            latents = _crop_latents(latents, crop_target)
            for i in range(batch):
                sample_idx = generated + i
                latent_path = latent_dir / f"latent_{sample_idx:04d}.pt"
                torch.save(latents[i].detach().cpu(), latent_path)
            generated += batch

    accelerator.wait_for_everyone()
    del edm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decode_device = (
        accelerator.device
        if args.decode_device is None and torch.cuda.is_available()
        else torch.device(args.decode_device or "cpu")
    )
    if accelerator.is_main_process:
        logger.info("Decoding latents with VAE on %s.", decode_device)

    if has_work:
        vae_ckpt = _load_checkpoint(Path(args.vae_checkpoint), device, VAE3DConfig)
        vae_cfg = vae_ckpt["config"]
        vae = VAE3D(vae_cfg)
        vae_state = _clean_state_dict(vae_ckpt["model_state_dict"])
        vae.load_state_dict(vae_state)
        vae.eval()
        vae.to(decode_device)
        vae_mean = vae_mean_cpu.to(decode_device)
        vae_std = vae_std_cpu.to(decode_device)

        generated = start_idx
        while generated < end_idx:
            batch = min(batch_size, end_idx - generated)
            latent_batch = []
            for i in range(batch):
                sample_idx = generated + i
                latent_path = latent_dir / f"latent_{sample_idx:04d}.pt"
                latent_batch.append(torch.load(latent_path, map_location=decode_device))
            latents = torch.stack(latent_batch, dim=0).to(decode_device)
            with torch.no_grad():
                with accelerator.autocast():
                    decoded = vae.decode(latents)
                # Denormalize VAE output back to original data space
                decoded = decoded * vae_std + vae_mean

            decoded_np = decoded.detach().cpu().float().numpy()
            for i in range(batch):
                sample_idx = generated + i
                output_path = output_dir / f"sample_{sample_idx:04d}.png"
                _plot_sample_slices(
                    decoded_np[i], output_path, z_indices, channel_names=channel_names
                )
            generated += batch
            del latents, decoded, decoded_np
            if decode_device.type == "cuda":
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saved %d samples to %s", total, output_dir)


if __name__ == "__main__":
    main()
