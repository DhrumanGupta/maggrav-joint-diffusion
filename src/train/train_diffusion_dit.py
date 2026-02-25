"""
Train a 3D Diffusion Transformer (DiT) directly on raw volumetric data (Density/Susceptibility).

Uses:
- Raw 2x200^3 input (no VAE).
- EDM Preconditioning and Loss.
- DiT3D architecture with learnable positional embeddings.
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

import wandb
from ..models.dit import DiT3D, DiT3DConfig
from ..models.edm import EDMPrecond, edm_loss, sample_edm_heun
from ..utils.datasets import JointDensitySuscDataset
from ..data.stats_utils import compute_streaming_stats_ddp, load_stats, save_stats

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        description="Train 3D DiT on raw density/susceptibility data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_diffusion_dit.yaml",
        help="Path to YAML config file.",
    )
    cli_args = parser.parse_args()
    config = load_config(cli_args.config)
    return argparse.Namespace(**config)


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

    # 1. Dataset
    # Raw data: (2, 200, 200, 200)
    dataset = JointDensitySuscDataset(args.zarr_path, return_index=False)
    
    # 2. Statistics (Mean/Std) for normalization
    stats_path = (
        Path(args.stats_path)
        if args.stats_path
        else output_dir / "dataset_stats.json"
    )

    if stats_path.exists() and not args.recompute_stats:
        if accelerator.is_main_process:
            logger.info("Loading dataset stats from %s", stats_path)
    else:
        if accelerator.is_main_process:
            logger.info("Computing dataset mean/std (distributed streaming)")

        stats = compute_streaming_stats_ddp(
            dataset,
            batch_size=args.stats_batch_size,
            num_workers=args.num_workers,
            accelerator=accelerator,
            num_channels=2,
        )

        if accelerator.is_main_process:
            save_stats(stats_path, stats)
            logger.info("Saved stats to %s", stats_path)
            
    accelerator.wait_for_everyone()
    stats = load_stats(stats_path)
    
    # Values for normalization: (1, 2, 1, 1, 1)
    data_mean = stats["mean"].float().view(1, 2, 1, 1, 1).to(accelerator.device)
    data_std = stats["std"].float().view(1, 2, 1, 1, 1).to(accelerator.device)

    if accelerator.is_main_process:
        logger.info("Data Stats - Mean: %s, Std: %s", stats["mean"], stats["std"])

    # 3. Data Loaders
    dataset_size = len(dataset)
    # Simple holdout (just last 1%) if needed, or use full. 
    # Let's do 1% holdout for consistency.
    holdout_size = max(1, math.ceil(dataset_size * 0.01))
    train_size = dataset_size - holdout_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=4, persistent_workers=True)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    # 4. Model
    dit_config = DiT3DConfig(
        input_size=tuple(args.input_size),
        patch_size=tuple(args.patch_size),
        patch_stride=tuple(getattr(args, "patch_stride", args.patch_size)),
        in_channels=2,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        learn_sigma=args.learn_sigma,
        dropout=args.dropout,
        overlap_norm=getattr(args, "overlap_norm", True),
        attn_backend=getattr(args, "attn_backend", "auto"),
    )
    raw_model = DiT3D(dit_config)
    
    # EDM Wrapper
    model = EDMPrecond(raw_model, sigma_data=args.sigma_data)
    
    # EMA
    ema_model = EMAModel(model.parameters(), decay=0.9999)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    # Scheduler
    # Use global batch size so warmup/decay are based on total sample counts.
    effective_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    total_optimizer_steps = args.max_steps // effective_batch_size
    warmup_steps = args.warmup_samples // effective_batch_size

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
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
    ema_model.to(accelerator.device)

    wandb_run = None
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters())
        logger.info("Model Parameters: %d", params)
        
        wandb_run = wandb.init(
            project="maggrav-dit",
            name=args.run_name,
            config=vars(args)
        )

    # 5. Loop
    global_samples = 0
    log_samples = 0
    log_loss_sum = 0.0
    
    next_log = args.log_every
    next_eval = args.eval_every_steps
    next_save = args.save_every
    
    data_iter = iter(train_loader)

    while global_samples < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        model.train()
        with accelerator.accumulate(model):
            batch = batch.to(accelerator.device, non_blocking=True)
            # Normalize
            batch = (batch - data_mean) / data_std
            
            loss = edm_loss(
                model, 
                batch, 
                mask=None, # No mask needed for raw data (fully valid 200^3)
                P_mean=args.P_mean,
                P_std=args.P_std,
                sigma_data=args.sigma_data
            )
            
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

        # Logging logic
        batch_size = batch.size(0)
        batch_size_tensor = torch.tensor(batch_size, device=accelerator.device)
        step_samples = batch_size * accelerator.num_processes
        global_samples += step_samples
        
        loss_sum = accelerator.reduce(
            loss.detach() * batch_size_tensor, reduction="sum"
        ).item()
        
        log_samples += step_samples
        log_loss_sum += loss_sum
        

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

        # Evaluation
        if global_samples >= next_eval:
            if accelerator.is_main_process:
                model_eval = accelerator.unwrap_model(model)
                ema_model.store(model_eval.parameters())
                ema_model.copy_to(model_eval.parameters())
                
                samples = sample_edm_heun(
                    model_eval,
                    num_samples=args.num_eval_samples,
                    latent_shape=(2, 200, 200, 200),
                    num_steps=args.num_eval_steps,
                    device=accelerator.device,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    rho=args.rho,
                    mask=None
                )
                
                ema_model.restore(model_eval.parameters())
                
                # Denormalize samples back to original space
                samples = samples * data_std + data_mean
                
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

        # Checkpoint
        if global_samples >= next_save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = output_dir / f"dit_ckpt_{global_samples}.pt"
                torch.save({
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "mean": stats["mean"],
                    "std": stats["std"]
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            next_save += args.save_every


if __name__ == "__main__":
    main()
