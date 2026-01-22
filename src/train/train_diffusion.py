"""
Train a 3D diffusion model on VAE latents with EDM (Elucidating the Design Space
of Diffusion-Based Generative Models).

Uses Hugging Face Accelerate for multi-GPU training and streams latents from a
Zarr store created by `encode_latents.py`.

Usage:
    python src/train_diffusion.py --config config/train_diffusion_edm.yaml
    accelerate launch src/train_diffusion.py --config config/train_diffusion_edm.yaml
"""

import warnings

warnings.filterwarnings("ignore", message="Profiler function")

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import zarr
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import wandb
from data.stats_utils import compute_latent_stats_ddp, load_stats, save_stats
from models.unet import UNet3DConfig, UNet3DDiffusion

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
        c_noise = sigma.squeeze().log() / 4  # (B,) for time embedding

        # Apply preconditioning
        scaled_input = c_in * x
        F_out = self.model(scaled_input, c_noise)
        denoised = c_skip * x + c_out * F_out

        return denoised


# -------------------------
# Dataset
# -------------------------


class LatentZarrDataset(Dataset):
    """Dataset that samples latents from a Zarr store on each access."""

    def __init__(self, zarr_path: str, pad_to: int = 32, use_logvar: bool = True):
        self.zarr_path = zarr_path
        self.pad_to = max(pad_to, 1)

        group = zarr.open_group(zarr_path, mode="r")
        if "latent_mu" not in group:
            raise ValueError(f"Zarr store missing 'latent_mu': {zarr_path}")

        mu_store = group["latent_mu"]
        self.length = mu_store.shape[0]
        self.channels = mu_store.shape[1]
        self.original_spatial = mu_store.shape[2:]
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


def create_padding_mask(
    padded_shape: Tuple[int, int, int, int],
    original_spatial: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a mask that is 1 in the valid (original) region and 0 in the padded region.

    Args:
        padded_shape: (C, D, H, W) - the padded tensor shape
        original_spatial: (D, H, W) - the original spatial dimensions before padding
        device: torch device

    Returns:
        Mask tensor of shape (1, 1, D, H, W) broadcastable to (B, C, D, H, W)
    """
    _, d_pad, h_pad, w_pad = padded_shape
    d_orig, h_orig, w_orig = original_spatial

    mask = torch.zeros(1, 1, d_pad, h_pad, w_pad, device=device)
    mask[:, :, :d_orig, :h_orig, :w_orig] = 1.0
    return mask


# -------------------------
# EDM Loss
# -------------------------


def edm_loss(
    model: EDMPrecond,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_data: float = 0.5,
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
        # Masked MSE loss: average over valid elements only
        diff_sq = (denoised - x) ** 2
        weighted_diff_sq = weight * diff_sq * mask
        loss = weighted_diff_sq.sum() / mask.sum() / x.shape[1]  # normalize by channels
    else:
        diff_sq = (denoised - x) ** 2
        loss = (weight * diff_sq).mean()

    return loss


# -------------------------
# EDM Sampler (Heun's 2nd-order)
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
    device: torch.device,
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
        denoised = model(x, sigma_batch)

        # Derivative: d = (x - D(x, σ)) / σ
        d_cur = (x - denoised) / sigma_cur

        # Euler step (predictor)
        x_next = x + (sigma_next - sigma_cur) * d_cur

        # Heun correction (if not at final step)
        if sigma_next > 0:
            sigma_batch_next = sigma_next.expand(num_samples)
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
# Config and Args
# -------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D diffusion model with EDM on latent Zarr."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_diffusion_edm.yaml",
        help="Path to YAML config file.",
    )
    cli_args = parser.parse_args()

    # Load config from YAML (single source of truth)
    config = load_config(cli_args.config)
    return argparse.Namespace(**config)


def _infer_latent_config(dataset: LatentZarrDataset) -> Tuple[int, int]:
    channels, depth, height, width = dataset.latent_shape
    if not (depth == height == width):
        raise ValueError(
            f"UNet3DConfig requires cubic inputs; got D/H/W = {depth}/{height}/{width}."
        )
    return channels, depth


# -------------------------
# Main Training Loop
# -------------------------


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    pad_to = max(args.pad_to, 1)
    dataset = LatentZarrDataset(
        args.latents_zarr,
        pad_to=pad_to,
        use_logvar=not args.no_logvar,
    )
    latent_channels, latent_spatial = _infer_latent_config(dataset)

    # Compute or load latent statistics for normalization
    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else output_dir / "latent_stats.json"
    )

    # Create padding mask for stats computation (exclude padded regions)
    stats_mask = None
    if dataset.pad_amounts != (0, 0, 0):
        stats_mask = create_padding_mask(
            padded_shape=dataset.latent_shape,
            original_spatial=dataset.original_spatial,
            device=torch.device(
                "cpu"
            ),  # Will be moved to device in compute_latent_stats_ddp
        )

    if stats_path.exists() and not args.recompute_stats:
        if accelerator.is_main_process:
            logger.info("Loading latent stats from %s", stats_path)
    else:
        if accelerator.is_main_process:
            logger.info("Computing latent mean/std (distributed streaming)")
        # Compute stats accounting for VAE sampling variance
        stats = compute_latent_stats_ddp(
            args.latents_zarr,
            args.stats_batch_size,
            args.num_workers,
            accelerator,
            num_channels=latent_channels,
            pad_to=pad_to,
            mask=stats_mask,
        )
        if accelerator.is_main_process:
            save_stats(stats_path, stats)
            logger.info("Saved latent stats to %s", stats_path)

    accelerator.wait_for_everyone()
    stats = load_stats(stats_path)

    # Reshape mean/std for broadcasting: (1, C, 1, 1, 1)
    latent_mean = stats["mean"].float().view(1, latent_channels, 1, 1, 1)
    latent_std = stats["std"].float().view(1, latent_channels, 1, 1, 1)

    if accelerator.is_main_process:
        logger.info(
            "Latent stats - mean: %s, std: %s",
            stats["mean"].tolist(),
            stats["std"].tolist(),
        )

    dataset_size = len(dataset)
    holdout_size = max(1, math.ceil(dataset_size * 0.01))
    train_size = dataset_size - holdout_size
    if train_size <= 0:
        raise ValueError(
            "Zarr store must contain more than 1 sample to reserve 1% for holdout."
        )

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    model_cfg = UNet3DConfig(
        in_channels=latent_channels,
        out_channels=latent_channels,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attn_levels=tuple(args.attn_levels),
        num_heads=args.num_heads,
        dropout=args.dropout,
        input_spatial=latent_spatial,
    )
    raw_model = UNet3DDiffusion(model_cfg)

    # Wrap with EDM preconditioning
    model = EDMPrecond(raw_model, sigma_data=args.sigma_data)

    # Create EMA model for improved sample quality
    ema_model = EMAModel(model.parameters(), decay=0.9999)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Calculate optimizer steps from sample counts
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    total_optimizer_steps = args.max_steps // effective_batch_size
    warmup_steps = args.warmup_samples // effective_batch_size

    # Create LR scheduler: linear warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_optimizer_steps - warmup_steps),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Move EMA to accelerator device after model is prepared
    ema_model.to(accelerator.device)

    # Move latent stats to device for normalization
    latent_mean = latent_mean.to(accelerator.device)
    latent_std = latent_std.to(accelerator.device)

    # Create padding mask if there is padding (to keep padded regions at zero during training)
    padding_mask = None
    if dataset.pad_amounts != (0, 0, 0):
        padding_mask = create_padding_mask(
            padded_shape=dataset.latent_shape,
            original_spatial=dataset.original_spatial,
            device=accelerator.device,
        )
        if accelerator.is_main_process:
            logger.info(
                "Padding mask created: original %s -> padded %s",
                dataset.original_spatial,
                dataset.padded_spatial,
            )

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model parameters: %d (trainable: %d)", total_params, trainable_params
        )

    wandb_run = None
    if accelerator.is_main_process:
        init_kwargs = {
            "project": "maggrav-edm",
            "config": {
                **vars(args),
                "latent_channels": latent_channels,
                "latent_spatial": latent_spatial,
                "dataset_size": dataset_size,
                "train_size": train_size,
                "holdout_size": holdout_size,
            },
        }
        if args.run_name:
            init_kwargs["name"] = args.run_name
        wandb_run = wandb.init(**init_kwargs)

    if accelerator.is_main_process:
        logger.info("Training for %d samples", args.max_steps)

    global_samples = 0
    log_samples = 0
    log_loss_sum = 0.0

    next_log = args.log_every
    next_eval = args.eval_every_steps
    next_save = args.save_every
    last_saved_step = 0

    epoch = 0
    data_iter = iter(train_loader)
    progress_bar = None
    if args.progress and accelerator.is_local_main_process:
        progress_bar = tqdm(total=args.max_steps, desc="Training", leave=True)

    while global_samples < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            continue

        model.train()
        with accelerator.accumulate(model):
            batch = batch.to(accelerator.device, non_blocking=True)
            # Normalize latents to zero mean, unit variance
            batch = (batch - latent_mean) / latent_std
            loss = edm_loss(
                model,
                batch,
                mask=padding_mask,
                P_mean=args.P_mean,
                P_std=args.P_std,
                sigma_data=args.sigma_data,
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update EMA weights
            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

        batch_size = batch.size(0)
        batch_size_tensor = torch.tensor(batch_size, device=accelerator.device)
        step_samples = accelerator.reduce(batch_size_tensor, reduction="sum")
        step_samples_int = int(step_samples.item())
        global_samples += step_samples_int

        loss_sum = accelerator.reduce(
            loss.detach() * batch_size_tensor, reduction="sum"
        )

        log_samples += step_samples_int
        log_loss_sum += loss_sum.item()

        if progress_bar is not None:
            progress_bar.update(step_samples_int)

        if global_samples >= next_log:
            if log_samples > 0 and accelerator.is_main_process:
                logger.info(
                    "Samples %d | Train Loss %.6f",
                    global_samples,
                    log_loss_sum / log_samples,
                )
            if wandb_run is not None and log_samples > 0:
                wandb.log(
                    {
                        "train/loss": log_loss_sum / log_samples,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_samples,
                )
            log_samples = 0
            log_loss_sum = 0.0
            next_log += args.log_every

        if global_samples >= next_eval:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                # Use EMA weights for evaluation sampling
                ema_model.store(unwrapped_model.parameters())  # Backup original weights
                ema_model.copy_to(unwrapped_model.parameters())  # Load EMA weights
                samples = sample_edm_heun(
                    unwrapped_model,
                    num_samples=args.num_eval_samples,
                    latent_shape=dataset.latent_shape,
                    num_steps=args.num_eval_steps,
                    device=accelerator.device,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    rho=args.rho,
                    mask=padding_mask,
                )
                ema_model.restore(
                    unwrapped_model.parameters()
                )  # Restore original weights
                # Denormalize samples back to original latent space
                samples = samples * latent_std + latent_mean
                orig_d, orig_h, orig_w = dataset.original_spatial
                if (
                    samples.shape[2] != orig_d
                    or samples.shape[3] != orig_h
                    or samples.shape[4] != orig_w
                ):
                    samples = samples[:, :, :orig_d, :orig_h, :orig_w]
                mean_map = samples.mean(dim=0)
                std_map = samples.std(dim=0)

                channels = mean_map.shape[0]
                mid = mean_map.shape[1] // 2
                for stat_name, stat_tensor in (("mean", mean_map), ("std", std_map)):
                    fig, axes = plt.subplots(
                        channels, 3, figsize=(12, 4 * channels), squeeze=False
                    )
                    for c in range(channels):
                        volume = stat_tensor[c].detach().cpu().numpy()
                        slices = [
                            ("XY (z={})".format(mid), volume[mid, :, :]),
                            ("XZ (y={})".format(mid), volume[:, mid, :]),
                            ("YZ (x={})".format(mid), volume[:, :, mid]),
                        ]
                        for j, (title, img) in enumerate(slices):
                            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                            ax = axes[c, j]
                            im = ax.imshow(img, cmap="viridis", aspect="equal")
                            ax.set_title(f"Ch {c} {stat_name}: {title}", fontsize=10)
                            ax.axis("off")
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    fig_path = (
                        output_dir / f"eval_{stat_name}_step_{global_samples}.png"
                    )
                    plt.savefig(
                        fig_path, dpi=150, bbox_inches="tight", facecolor="white"
                    )
                    if wandb_run is not None:
                        wandb.log(
                            {f"eval/{stat_name}": wandb.Image(fig)},
                            step=global_samples,
                        )
                    plt.close(fig)

            next_eval += args.eval_every_steps

        if global_samples >= next_save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    "global_samples": global_samples,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                    "args": vars(args),
                    "latents_zarr": args.latents_zarr,
                    "original_spatial": dataset.original_spatial,
                    "latent_shape": dataset.latent_shape,
                    "sigma_data": args.sigma_data,
                    "latent_mean": latent_mean.cpu(),
                    "latent_std": latent_std.cpu(),
                }
                checkpoint_path = (
                    output_dir / f"edm_checkpoint_step_{global_samples}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info("Saved checkpoint: %s", checkpoint_path)
                last_saved_step = global_samples
            next_save += args.save_every

    if progress_bar is not None:
        progress_bar.close()

    if last_saved_step != global_samples:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                "global_samples": global_samples,
                "model_state_dict": unwrapped_model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_cfg,
                "args": vars(args),
                "latents_zarr": args.latents_zarr,
                "original_spatial": dataset.original_spatial,
                "latent_shape": dataset.latent_shape,
                "sigma_data": args.sigma_data,
                "latent_mean": latent_mean.cpu(),
                "latent_std": latent_std.cpu(),
            }
            checkpoint_path = output_dir / f"edm_checkpoint_step_{global_samples}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
