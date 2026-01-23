"""
Train a 3D VAE on density + susceptibility volumes.

Usage:
    python src/train_vae.py --config config/train_vae.yaml
    accelerate launch src/train_vae.py --config config/train_vae.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import wandb
from ..data.stats_utils import compute_streaming_stats_ddp, load_stats, save_stats
from ..models.vae import VAE3D, VAE3DConfig
from ..utils.datasets import JointDensitySuscDataset

# Padding constants: original data is 200³, pad to 256³ for nice latent dimensions
ORIGINAL_SIZE = 200
PADDED_SIZE = 256
PAD_AMOUNT = (PADDED_SIZE - ORIGINAL_SIZE) // 2  # 28 on each side
DOWNSAMPLE_FACTOR = 8  # Must match config
# Latent padding: use ceil to ensure we only include fully-valid latent positions
LATENT_PAD = (PAD_AMOUNT + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR  # 4
LATENT_VALID_SIZE = (PADDED_SIZE // DOWNSAMPLE_FACTOR) - 2 * LATENT_PAD  # 24

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def pad_to_256(x: torch.Tensor) -> torch.Tensor:
    """Pad input from (B, C, 200, 200, 200) to (B, C, 256, 256, 256) with zeros."""
    # F.pad expects (left, right, top, bottom, front, back) for 3D
    return F.pad(
        x, (PAD_AMOUNT, PAD_AMOUNT, PAD_AMOUNT, PAD_AMOUNT, PAD_AMOUNT, PAD_AMOUNT)
    )


def extract_valid_region(x: torch.Tensor) -> torch.Tensor:
    """Extract the original 200³ region from a 256³ tensor."""
    return x[
        :,
        :,
        PAD_AMOUNT : PAD_AMOUNT + ORIGINAL_SIZE,
        PAD_AMOUNT : PAD_AMOUNT + ORIGINAL_SIZE,
        PAD_AMOUNT : PAD_AMOUNT + ORIGINAL_SIZE,
    ]


def masked_mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss only on the valid 200³ region."""
    recon_valid = extract_valid_region(recon)
    target_valid = extract_valid_region(target)
    return F.mse_loss(recon_valid, target_valid)


def extract_valid_latent_region(x: torch.Tensor) -> torch.Tensor:
    """Extract the latent region corresponding to valid input data.

    With 8x downsampling and 28-pixel padding, the valid latent region
    is [4:-4] in each spatial dimension (24³ out of 32³).
    """
    return x[
        :, :, LATENT_PAD:-LATENT_PAD, LATENT_PAD:-LATENT_PAD, LATENT_PAD:-LATENT_PAD
    ]


def masked_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL loss only on the valid latent region."""
    mu_valid = extract_valid_latent_region(mu)
    logvar_valid = extract_valid_latent_region(logvar)
    return -0.5 * torch.mean(1 + logvar_valid - mu_valid.pow(2) - logvar_valid.exp())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    kl_weight: float,
    accelerator: Accelerator,
) -> Dict[str, float]:
    model.eval()
    device = accelerator.device

    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        # Normalize then pad to 256³
        batch = (batch - mean) / std
        batch_padded = pad_to_256(batch)

        with accelerator.autocast():
            recon, mu, logvar = model(batch_padded)
            # Compute losses only on valid regions
            recon_loss = masked_mse_loss(recon, batch_padded)
            kl_loss = masked_kl_loss(mu, logvar)
            loss = recon_loss + kl_weight * kl_loss

        batch_size = batch.size(0)
        total_samples += batch_size
        total_recon += recon_loss.detach() * batch_size
        total_kl += kl_loss.detach() * batch_size
        total_loss += loss.detach() * batch_size

    total_recon = accelerator.reduce(total_recon, reduction="sum")
    total_kl = accelerator.reduce(total_kl, reduction="sum")
    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_samples = accelerator.reduce(total_samples, reduction="sum")

    if total_samples.item() == 0:
        return {"recon_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0}

    return {
        "recon_loss": (total_recon / total_samples).item(),
        "kl_loss": (total_kl / total_samples).item(),
        "total_loss": (total_loss / total_samples).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D VAE with Accelerate")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_vae.yaml",
        help="Path to YAML config file.",
    )
    # CLI overrides for sweep/grid search
    parser.add_argument(
        "--kl_weight", type=float, default=None, help="Override kl_weight from config"
    )
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping/dict: {cli_args.config}")

    # Apply CLI overrides
    if cli_args.kl_weight is not None:
        config["kl_weight"] = cli_args.kl_weight

    return argparse.Namespace(**config)


def main() -> None:
    args = parse_args()
    if not hasattr(args, "kl_warmup_steps"):
        args.kl_warmup_steps = 0
    if not hasattr(args, "warmup_steps"):
        args.warmup_steps = 0

    # Initialize CUDA context BEFORE creating Accelerator to avoid NCCL errors
    # This must happen before any distributed initialization
    if torch.cuda.is_available():
        # Get local rank from environment (set by accelerate/torchrun)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.cuda.init()
        torch.cuda.synchronize()
        # Force CUDA context creation
        _ = torch.empty(1, device=f"cuda:{local_rank}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else output_dir / "vae_stats.json"
    )

    if stats_path.exists() and not args.recompute_stats:
        if accelerator.is_main_process:
            logger.info(f"Loading stats from {stats_path}")
    else:
        logger.info("Computing global mean/std (distributed streaming)")
        stats_dataset = JointDensitySuscDataset(args.zarr_path)
        stats = compute_streaming_stats_ddp(
            stats_dataset,
            args.stats_batch_size,
            args.num_workers,
            accelerator,
            num_channels=2,  # density + susceptibility
        )
        if accelerator.is_main_process:
            save_stats(stats_path, stats)
            logger.info(f"Saved stats to {stats_path}")

    accelerator.wait_for_everyone()
    stats = load_stats(stats_path)

    mean = stats["mean"].float().view(1, 2, 1, 1, 1)
    std = stats["std"].float().view(1, 2, 1, 1, 1)

    dataset = JointDensitySuscDataset(args.zarr_path)
    dataset_size = len(dataset)
    train_size = int(0.98 * dataset_size)
    val_size = int(0.005 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_subset = Subset(dataset, range(0, train_size))
    val_subset = Subset(dataset, range(train_size, train_size + val_size))
    test_subset = Subset(
        dataset,
        range(train_size + val_size, dataset_size),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        logger.info(
            "Dataset split sizes - train: %d | val: %d | test: %d",
            train_size,
            val_size,
            test_size,
        )

    latent_channels = (
        args.latent_channels if args.latent_channels is not None else args.base_channels
    )
    model_cfg = VAE3DConfig(
        in_channels=2,
        base_channels=args.base_channels,
        latent_channels=latent_channels,
        downsample_factor=args.downsample_factor,
        blocks_per_stage=args.blocks_per_stage,
    )
    model = VAE3D(model_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # Print number of parameters
    if accelerator.is_main_process:
        logger.info(
            f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)

    wandb_run = None
    if accelerator.is_main_process:
        init_kwargs = {
            "project": os.getenv("WANDB_PROJECT", "maggrav-vae"),
            "config": {
                **vars(args),
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
            },
        }
        run_name = os.getenv("WANDB_RUN_NAME")
        if run_name:
            init_kwargs["name"] = run_name
        wandb_run = wandb.init(**init_kwargs)

    if accelerator.is_main_process:
        logger.info("Training for %d samples", args.max_steps)

    global_samples = 0
    best_val_loss = float("inf")
    best_path = output_dir / "vae_best.json"
    log_samples = 0
    log_total_sum = 0.0
    log_recon_sum = 0.0
    log_kl_sum = 0.0
    eval_samples = 0
    eval_total_sum = 0.0
    eval_recon_sum = 0.0
    eval_kl_sum = 0.0

    next_log = args.log_every
    next_eval = args.eval_every_steps
    next_save = args.save_every
    last_saved_step = 0

    epoch = 0
    data_iter = iter(train_loader)
    progress_bar = None
    if args.progress and accelerator.is_local_main_process:
        progress_bar = tqdm(total=args.max_steps, desc="Training", leave=True)

    profile_window = args.profile and accelerator.is_main_process
    window_data_time = 0.0
    window_compute_time = 0.0
    window_steps = 0
    window_start = time.perf_counter() if profile_window else None

    while global_samples < args.max_steps:
        if profile_window:
            batch_wait_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            continue
        if profile_window:
            window_data_time += time.perf_counter() - batch_wait_start
            compute_start = time.perf_counter()

        model.train()
        lr_warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
        if lr_warmup_steps > 0:
            lr_scale = min(1.0, float(global_samples) / float(lr_warmup_steps))
            current_lr = float(args.lr) * lr_scale
        else:
            current_lr = float(args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        kl_warmup_steps = int(getattr(args, "kl_warmup_steps", 0) or 0)
        if kl_warmup_steps > 0:
            kl_weight = float(args.kl_weight) * min(
                1.0, float(global_samples) / float(kl_warmup_steps)
            )
        else:
            kl_weight = float(args.kl_weight)
        with accelerator.accumulate(model):
            batch = batch.to(accelerator.device, non_blocking=True)
            # Normalize then pad to 256³
            batch = (batch - mean) / std
            batch_padded = pad_to_256(batch)

            with accelerator.autocast():
                recon, mu, logvar = model(batch_padded)
                # Compute losses only on valid regions
                recon_loss = masked_mse_loss(recon, batch_padded)
                kl_loss = masked_kl_loss(mu, logvar)
                loss = recon_loss + kl_weight * kl_loss

            accelerator.backward(loss)
            if accelerator.sync_gradients and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        batch_size = batch.size(0)
        batch_size_tensor = torch.tensor(batch_size, device=accelerator.device)
        step_samples = accelerator.reduce(batch_size_tensor, reduction="sum")
        step_samples_int = int(step_samples.item())
        global_samples += step_samples_int

        loss_sum = accelerator.reduce(
            loss.detach() * batch_size_tensor, reduction="sum"
        )
        recon_sum = accelerator.reduce(
            recon_loss.detach() * batch_size_tensor, reduction="sum"
        )
        kl_sum = accelerator.reduce(
            kl_loss.detach() * batch_size_tensor, reduction="sum"
        )

        log_samples += step_samples_int
        log_total_sum += loss_sum.item()
        log_recon_sum += recon_sum.item()
        log_kl_sum += kl_sum.item()

        eval_samples += step_samples_int
        eval_total_sum += loss_sum.item()
        eval_recon_sum += recon_sum.item()
        eval_kl_sum += kl_sum.item()

        if progress_bar is not None:
            progress_bar.update(step_samples_int)
        window_steps += 1

        if global_samples >= next_log:
            if profile_window and window_start is not None and window_steps > 0:
                window_duration = time.perf_counter() - window_start
                data_pct = (
                    (window_data_time / window_duration) * 100.0
                    if window_duration > 0
                    else 0.0
                )
                compute_pct = (
                    (window_compute_time / window_duration) * 100.0
                    if window_duration > 0
                    else 0.0
                )
                logger.info(
                    "Profile window: loop=%.3fs | data=%.3fs (%.1f%%) | compute=%.3fs (%.1f%%) | steps=%d",
                    window_duration,
                    window_data_time,
                    data_pct,
                    window_compute_time,
                    compute_pct,
                    window_steps,
                )
                window_data_time = 0.0
                window_compute_time = 0.0
                window_steps = 0
                window_start = time.perf_counter()
            if log_samples > 0 and accelerator.is_main_process:
                logger.info(
                    "Samples %d | Train Loss %.6f | Recon %.6f | KL %.6f",
                    global_samples,
                    log_total_sum / log_samples,
                    log_recon_sum / log_samples,
                    log_kl_sum / log_samples,
                )
            if wandb_run is not None and log_samples > 0:
                wandb.log(
                    {
                        "train/recon_loss": log_recon_sum / log_samples,
                        "train/kl_loss": log_kl_sum / log_samples,
                        "train/total_loss": log_total_sum / log_samples,
                        "train/kl_weight": kl_weight,
                        "train/lr": current_lr,
                    },
                    step=global_samples,
                )
            log_samples = 0
            log_total_sum = 0.0
            log_recon_sum = 0.0
            log_kl_sum = 0.0
            next_log += args.log_every
        if profile_window:
            window_compute_time += time.perf_counter() - compute_start

        if global_samples >= next_eval:
            val_metrics = evaluate(model, val_loader, mean, std, kl_weight, accelerator)
            if accelerator.is_main_process:
                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    payload = {
                        "best_val_loss": best_val_loss,
                        "global_samples": global_samples,
                        "checkpoint": f"vae_checkpoint_step_{global_samples}.pt",
                    }
                    with best_path.open("w") as f:
                        json.dump(payload, f, indent=2)
                    logger.info("Updated best checkpoint: %s", best_path)
                if eval_samples > 0:
                    logger.info(
                        "Samples %d | Train Loss %.6f | Recon %.6f | KL %.6f",
                        global_samples,
                        eval_total_sum / eval_samples,
                        eval_recon_sum / eval_samples,
                        eval_kl_sum / eval_samples,
                    )
                logger.info(
                    "Samples %d | Val Loss %.6f | Recon %.6f | KL %.6f",
                    global_samples,
                    val_metrics["total_loss"],
                    val_metrics["recon_loss"],
                    val_metrics["kl_loss"],
                )
            if wandb_run is not None:
                payload = {
                    "val/recon_loss": val_metrics["recon_loss"],
                    "val/kl_loss": val_metrics["kl_loss"],
                    "val/total_loss": val_metrics["total_loss"],
                }
                if eval_samples > 0:
                    payload.update(
                        {
                            "train/recon_loss": eval_recon_sum / eval_samples,
                            "train/kl_loss": eval_kl_sum / eval_samples,
                            "train/total_loss": eval_total_sum / eval_samples,
                        }
                    )
                wandb.log(payload, step=global_samples)

            eval_samples = 0
            eval_total_sum = 0.0
            eval_recon_sum = 0.0
            eval_kl_sum = 0.0
            next_eval += args.eval_every_steps

        if global_samples >= next_save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    "global_samples": global_samples,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                    "args": vars(args),
                    "stats_path": str(stats_path),
                }
                checkpoint_path = (
                    output_dir / f"vae_checkpoint_step_{global_samples}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info("Saved checkpoint: %s", checkpoint_path)
                last_saved_step = global_samples
            next_save += args.save_every

    if progress_bar is not None:
        progress_bar.close()

    if profile_window and window_start is not None and window_steps > 0:
        window_duration = time.perf_counter() - window_start
        data_pct = (
            (window_data_time / window_duration) * 100.0 if window_duration > 0 else 0.0
        )
        compute_pct = (
            (window_compute_time / window_duration) * 100.0
            if window_duration > 0
            else 0.0
        )
        logger.info(
            "Profile window (partial): loop=%.3fs | data=%.3fs (%.1f%%) | compute=%.3fs (%.1f%%) | steps=%d",
            window_duration,
            window_data_time,
            data_pct,
            window_compute_time,
            compute_pct,
            window_steps,
        )

    if last_saved_step != global_samples:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                "global_samples": global_samples,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_cfg,
                "args": vars(args),
                "stats_path": str(stats_path),
            }
            checkpoint_path = output_dir / f"vae_checkpoint_step_{global_samples}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
