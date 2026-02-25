"""
Train a 3D diffusion model on VAE latents with rectified flow.

Uses Hugging Face Accelerate for multi-GPU training and streams latents from a
Zarr store created by `encode_latents.py`.

Usage:
    python src/train_flow.py --config config/train_flow.yaml
    accelerate launch src/train_flow.py --config config/train_flow.yaml
"""

import warnings

warnings.filterwarnings("ignore", message="Profiler function")

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

from ..data.stats_utils import compute_latent_stats_ddp, load_stats, save_stats
from ..models.flow import rectified_flow_loss, sample_rectified_flow
from ..models.unet import UNet3DConfig, UNet3DDiffusion
from ..utils.datasets import LatentZarrDataset
from ..utils.stats import get_effective_std, get_global_std

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OVERFIT_SAMPLES = True
NUM_SAMPLES_TO_OVERFIT = 10


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D diffusion model with rectified flow on latent Zarr."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_flow.yaml",
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


def _extract_latent_batch(batch_payload: Any) -> torch.Tensor:
    """Extract latent tensor from DataLoader batch payload.

    LatentZarrDataset returns (weight, latent) pairs, which the default
    DataLoader collate function materializes as a list/tuple of tensors.
    """
    if torch.is_tensor(batch_payload):
        return batch_payload

    if (
        isinstance(batch_payload, (list, tuple))
        and len(batch_payload) == 2
        and torch.is_tensor(batch_payload[1])
    ):
        return batch_payload[1]

    raise TypeError(
        "Expected batch payload to be a Tensor or (weights, latents) pair; "
        f"got {type(batch_payload)!r}."
    )


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

    use_logvar = not args.no_logvar
    dataset = LatentZarrDataset(
        args.latents_zarr,
        use_logvar=use_logvar,
    )
    latent_channels, latent_spatial = _infer_latent_config(dataset)

    # Compute or load latent statistics for normalization
    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else output_dir / "latent_stats.json"
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
        )
        if accelerator.is_main_process:
            save_stats(stats_path, stats)
            logger.info("Saved latent stats to %s", stats_path)

    accelerator.wait_for_everyone()
    stats = load_stats(stats_path)

    # Reshape mean/std for broadcasting: (1, C, 1, 1, 1)
    latent_mean = stats["mean"].float().view(1, latent_channels, 1, 1, 1)

    # Compute std: either global (single value) or per-channel
    if args.use_single_vae_scaling:
        # Use global std across all channels
        global_std = get_global_std(stats, use_logvar)
        latent_std = global_std.view(1, 1, 1, 1, 1).expand(1, latent_channels, 1, 1, 1)
    else:
        # Use per-channel std (current behavior)
        latent_std = get_effective_std(stats, use_logvar).view(
            1, latent_channels, 1, 1, 1
        )

    if accelerator.is_main_process:
        if args.use_single_vae_scaling:
            logger.info(
                "Latent stats - mean: %s, global std: %.6f (single scaling mode)",
                stats["mean"].tolist(),
                latent_std[0, 0, 0, 0, 0].item(),
            )
        else:
            logger.info(
                "Latent stats - mean: %s, std: %s (per-channel scaling)",
                stats["mean"].tolist(),
                stats["std"].tolist(),
            )

    dataset_size = len(dataset)
    holdout_size = max(1, math.ceil(dataset_size * 0.01))
    train_size = dataset_size - holdout_size
    if OVERFIT_SAMPLES:
        train_size = NUM_SAMPLES_TO_OVERFIT
    if train_size <= 0:
        raise ValueError(
            "Zarr store must contain more than 1 sample to reserve 1% for holdout."
        )

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))

    accelerator.wait_for_everyone()

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

    if accelerator.is_main_process and OVERFIT_SAMPLES:
        # Plot latent depth slices for each sample in the training split.
        latent_plots_dir = output_dir / "latents"
        latent_plots_dir.mkdir(parents=True, exist_ok=True)
        # Avoid mixing stale images from previous runs.
        for existing_plot in latent_plots_dir.glob("*_slices.png"):
            existing_plot.unlink()
        for sample_idx in range(len(train_dataset)):
            latent = train_dataset[sample_idx][1]
            if latent.ndim == 5:
                # If a batch axis is present, take the first sample.
                latent = latent[0]
            if latent.ndim != 4:
                raise ValueError(
                    f"Expected latent with shape [C, D, H, W], got {tuple(latent.shape)}"
                )

            latent_np = latent.detach().cpu().float().numpy()
            finite_ratio = float(np.isfinite(latent_np).mean())
            sample_std = float(np.nanstd(latent_np))
            if finite_ratio < 1.0 or sample_std == 0.0:
                logger.warning(
                    "Sample %d has finite_ratio=%.4f and std=%.6f; plots may look blank.",
                    sample_idx,
                    finite_ratio,
                    sample_std,
                )
            num_channels, depth, _, _ = latent_np.shape
            num_cols = 4
            num_rows = num_channels
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(12, 4 * num_rows), squeeze=False
            )
            for c in range(num_channels):
                for col_idx in range(num_cols):
                    # Spread sampled slices across available depth.
                    z = min(
                        depth - 1,
                        int(round(col_idx * (depth - 1) / max(1, num_cols - 1))),
                    )
                    ax = axes[c, col_idx]
                    slice_2d = latent_np[c, z, :, :]
                    finite_mask = np.isfinite(slice_2d)
                    if not finite_mask.any():
                        ax.set_facecolor("lightgray")
                        ax.text(
                            0.5,
                            0.5,
                            "all NaN/Inf",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
                    else:
                        # Use percentile scaling so low-variance slices remain visible.
                        vmin = float(np.nanpercentile(slice_2d, 1))
                        vmax = float(np.nanpercentile(slice_2d, 99))
                        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                            vmin = float(np.nanmin(slice_2d))
                            vmax = float(np.nanmax(slice_2d))
                            if vmin == vmax:
                                vmax = vmin + 1e-6
                        ax.imshow(slice_2d, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"ch{c} z={z}", fontsize=10)
                    ax.axis("off")
            plt.tight_layout()
            fig_path = latent_plots_dir / f"{sample_idx}_slices.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

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
    model = UNet3DDiffusion(model_cfg)

    # Create EMA model for improved sample quality
    ema_model = EMAModel(model.parameters(), decay=0 if OVERFIT_SAMPLES else 0.9995)

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

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model parameters: %d (trainable: %d)", total_params, trainable_params
        )

    wandb_run = None
    if accelerator.is_main_process:
        init_kwargs = {
            "project": "maggrav-rectified-flow",
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
            batch = _extract_latent_batch(batch)
            batch = batch.to(accelerator.device, non_blocking=True)
            # Normalize latents to zero mean, unit variance
            batch = (batch - latent_mean) / latent_std
            loss = rectified_flow_loss(model, batch)

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
                samples = sample_rectified_flow(
                    unwrapped_model,
                    num_samples=args.num_eval_samples,
                    latent_shape=dataset.latent_shape,
                    num_steps=args.num_eval_steps,
                    device=accelerator.device,
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

                first_sample = samples[0].detach().cpu().float().numpy()
                num_channels, depth, _, _ = first_sample.shape
                num_cols = 4
                num_rows = num_channels
                fig, axes = plt.subplots(
                    num_rows, num_cols, figsize=(12, 4 * num_rows), squeeze=False
                )
                for c in range(num_channels):
                    for i in range(num_cols):
                        z = min(
                            depth - 1,
                            int(round(i * (depth - 1) / max(1, num_cols - 1))),
                        )
                        ax = axes[c, i]
                        ax.imshow(first_sample[c, z, :, :], cmap="viridis")
                        ax.set_title(f"ch{c} z={z}", fontsize=10)
                        ax.axis("off")
                plt.tight_layout()
                fig_path = (
                    output_dir
                    / f"eval_first_generated_latent_step_{global_samples}.png"
                )
                plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
                if wandb_run is not None:
                    wandb.log(
                        {"eval/first_generated_latent": wandb.Image(fig)},
                        step=global_samples,
                    )
                plt.close(fig)

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
                    "latent_mean": latent_mean.cpu(),
                    "latent_std": latent_std.cpu(),
                    "use_single_vae_scaling": args.use_single_vae_scaling,
                }
                checkpoint_path = (
                    output_dir / f"flow_checkpoint_step_{global_samples}.pt"
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
                "latent_mean": latent_mean.cpu(),
                "latent_std": latent_std.cpu(),
                "use_single_vae_scaling": args.use_single_vae_scaling,
            }
            checkpoint_path = output_dir / f"flow_checkpoint_step_{global_samples}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
