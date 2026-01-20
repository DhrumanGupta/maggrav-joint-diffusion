"""
Train a 3D diffusion model on VAE latents with rectified flow.

Uses Hugging Face Accelerate for multi-GPU training and streams latents from a
Zarr store created by `encode_latents.py`.

Usage:
    python src/train_diffusion.py --latents_zarr /path/to/latents.zarr --max_steps 100000
    accelerate launch src/train_diffusion.py --latents_zarr /path/to/latents.zarr --max_steps 100000
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import wandb
from unet import UNet3DConfig, UNet3DDiffusion

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        mu = self._pad_tensor(mu)
        if self.use_logvar and self._logvar_store is not None:
            logvar = torch.as_tensor(self._logvar_store[idx], dtype=torch.float32)
            logvar = self._pad_tensor(logvar)
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps
        return mu

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_d, pad_h, pad_w = self.pad_amounts
        if pad_d == pad_h == pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))


def rectified_flow_loss(
    model: UNet3DDiffusion,
    x: torch.Tensor,
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
    """
    batch_size = x.shape[0]

    # Logit-normal sampling: sample from normal, then apply sigmoid
    u = torch.randn(batch_size, device=x.device) * logit_std + logit_mean
    t = torch.sigmoid(u)  # t in (0, 1), concentrated around 0.5

    z0 = torch.randn_like(x)
    t_bc = t[:, None, None, None, None]
    x_t = (1 - t_bc) * x + t_bc * z0
    v = z0 - x
    v_pred = model(x_t, t)
    return F.mse_loss(v_pred, v)


@torch.no_grad()
def sample_rectified_flow(
    model: UNet3DDiffusion,
    num_samples: int,
    latent_shape: Tuple[int, int, int, int],
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Basic Euler sampler for rectified flow.
    Starts from x(t=1) ~ N(0, I) and integrates to t=0.
    """
    model.eval()
    channels, depth, height, width = latent_shape
    x = torch.randn(
        num_samples, channels, depth, height, width, device=device, dtype=torch.float32
    )
    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t = t_vals[i]
        t_next = t_vals[i + 1]
        dt = t_next - t
        t_batch = torch.full((num_samples,), t, device=device)
        v = model(x, t_batch)
        x = x + v * dt
    model.train()
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D diffusion model with rectified flow on latent Zarr."
    )

    # Data
    parser.add_argument("--latents_zarr", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--no_logvar",
        action="store_true",
        help="Disable logvar sampling even if present in the Zarr store.",
    )
    parser.add_argument(
        "--pad_to",
        type=int,
        default=32,
        help="Pad each latent volume (D/H/W) up to at least this size.",
    )

    # Model
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument(
        "--channel_mults",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
    )
    parser.add_argument(
        "--attn_levels",
        type=int,
        nargs="+",
        default=(2, 3),
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    # Timestep sampling (logit-normal distribution)
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean of logit-normal distribution for timestep sampling.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std of logit-normal distribution for timestep sampling.",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--warmup_samples",
        type=int,
        default=100_000,
        help="Number of samples for linear LR warmup.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150_000_000,
        help="Total number of samples to process before stopping.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=250_000,
        help="Run validation every N samples processed.",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=32,
        help="Number of samples to generate during evaluation.",
    )
    parser.add_argument(
        "--num_eval_steps",
        type=int,
        default=64,
        help="Number of inference steps for rectified flow sampling.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/diffusion_bigger")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=250_000,
        help="Save checkpoint every N samples processed.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5_000,
        help="Log training metrics every N samples processed.",
    )
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


def _infer_latent_config(dataset: LatentZarrDataset) -> Tuple[int, int]:
    channels, depth, height, width = dataset.latent_shape
    if not (depth == height == width):
        raise ValueError(
            f"UNet3DConfig requires cubic inputs; got D/H/W = {depth}/{height}/{width}."
        )
    return channels, depth


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
    model = UNet3DDiffusion(model_cfg)

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

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %d", total_params)

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
            batch = batch.to(accelerator.device, non_blocking=True)
            loss = rectified_flow_loss(
                model, batch, logit_mean=args.logit_mean, logit_std=args.logit_std
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
                ema_model.copy_to(unwrapped_model.parameters())
                samples = sample_rectified_flow(
                    unwrapped_model,
                    num_samples=args.num_eval_samples,
                    latent_shape=dataset.latent_shape,
                    num_steps=args.num_eval_steps,
                    device=accelerator.device,
                )
                # Restore original weights after sampling
                ema_model.restore(unwrapped_model.parameters())
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
                }
                checkpoint_path = (
                    output_dir / f"diffusion_checkpoint_step_{global_samples}.pt"
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
            }
            checkpoint_path = (
                output_dir / f"diffusion_checkpoint_step_{global_samples}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
