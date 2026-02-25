import argparse
import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

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

from ..data.stats_utils import compute_latent_stats_ddp
from ..data.stats_utils import load_stats as load_latent_stats
from ..data.stats_utils import save_stats as save_latent_stats
from ..models.biflownet import BiFlowNet
from ..models.vae import VAE3D, VAE3DConfig
from ..models.vae_attention import VAE3DAttention, VAE3DAttentionConfig
from ..utils.checkpoint import clean_state_dict, load_checkpoint
from ..utils.datasets import LatentZarrDataset
from ..utils.latent_weighted_filtering import LatentWeightedFiltering
from ..utils.latent_weighting import LatentDensityKNNWeighter
from ..utils.masking import create_padding_mask
from ..utils.stats import get_effective_std, get_global_std
from ..utils.stats import load_stats as load_data_stats
from ..utils.stats import reshape_stats_for_broadcast

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OVERFIT_SAMPLES = False
NUM_SAMPLES_TO_OVERFIT = 10


@dataclass
class ObjectiveHooks:
    wrap_model: Callable[[BiFlowNet, argparse.Namespace], torch.nn.Module]
    compute_per_sample_loss: Callable[
        [torch.nn.Module, torch.Tensor, Optional[torch.Tensor], argparse.Namespace],
        torch.Tensor,
    ]
    sample_latents: Callable[
        [
            torch.nn.Module,
            int,
            Tuple[int, int, int, int],
            torch.device,
            Optional[torch.Tensor],
            Accelerator,
            argparse.Namespace,
        ],
        torch.Tensor,
    ]
    wandb_project: str
    checkpoint_prefix: str
    checkpoint_extras: Callable[[argparse.Namespace], Dict[str, Any]]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_config_args(description: str, default_config: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to YAML config file.",
    )
    cli_args = parser.parse_args()
    return argparse.Namespace(**load_config(cli_args.config))


def _infer_latent_config(dataset: LatentZarrDataset) -> Tuple[int, int]:
    channels, depth, height, width = dataset.latent_shape
    if not (depth == height == width):
        raise ValueError(
            f"BiFlowNet expects cubic inputs; got D/H/W = {depth}/{height}/{width}."
        )
    return channels, depth


def _to_3tuple(val: Any) -> Tuple[int, int, int]:
    if isinstance(val, (list, tuple)):
        if len(val) != 3:
            raise ValueError(f"Expected 3-tuple for sub_volume_size, got {val}.")
        return (int(val[0]), int(val[1]), int(val[2]))
    return (int(val), int(val), int(val))


def _build_latent_sample_filter(args: argparse.Namespace) -> Optional[Any]:
    cfg = getattr(args, "latent_weighing", None)
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("latent_weighing must be a mapping in config when provided.")

    cfg = dict(cfg)
    enabled = bool(cfg.pop("enabled", True))
    if not enabled:
        return None

    filter_type = str(cfg.pop("type", "density_weighting"))
    if filter_type == "density_weighting":
        return LatentDensityKNNWeighter(**cfg)
    if filter_type == "weighted_filtering":
        if "num_samples" not in cfg:
            raise ValueError(
                "latent_weighing.type=weighted_filtering requires num_samples"
            )
        num_samples = int(cfg.pop("num_samples"))
        return LatentWeightedFiltering(num_samples=num_samples, **cfg)

    raise ValueError(
        "latent_weighing.type must be 'density_weighting' or 'weighted_filtering', "
        f"got {filter_type!r}."
    )


def _center_crop(x: torch.Tensor, target_size: Optional[int]) -> torch.Tensor:
    if target_size is None:
        return x
    d, h, w = x.shape[2:]
    if target_size > d or target_size > h or target_size > w:
        raise ValueError(
            f"Target size {target_size} larger than tensor size {d}/{h}/{w}."
        )
    start_d = (d - target_size) // 2
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return x[
        :,
        :,
        start_d : start_d + target_size,
        start_h : start_h + target_size,
        start_w : start_w + target_size,
    ]


def _get_decoder_output_spatial(
    vae_cfg: Union[VAE3DConfig, VAE3DAttentionConfig],
    latent_shape: Tuple[int, ...],
) -> Tuple[int, int, int]:
    if isinstance(vae_cfg, VAE3DConfig):
        return tuple(s * vae_cfg.downsample_factor for s in latent_shape[1:4])
    if isinstance(vae_cfg, VAE3DAttentionConfig):
        return (200, 200, 200)
    raise ValueError(
        f"Unsupported VAE config type for decoder output spatial: {type(vae_cfg).__name__}"
    )


def _load_vae(checkpoint_path: Path, device: torch.device) -> tuple[
    Union[VAE3D, VAE3DAttention],
    Union[VAE3DConfig, VAE3DAttentionConfig],
    Optional[str],
]:
    ckpt = load_checkpoint(
        checkpoint_path, device, config_classes=[VAE3DConfig, VAE3DAttentionConfig]
    )
    if "config" not in ckpt:
        raise ValueError(f"VAE checkpoint missing config: {checkpoint_path}")
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        if "downsample_factor" in cfg:
            cfg = VAE3DConfig(**cfg)
        else:
            cfg = VAE3DAttentionConfig(**cfg)
    if isinstance(cfg, VAE3DConfig):
        model = VAE3D(cfg)
    elif isinstance(cfg, VAE3DAttentionConfig):
        model = VAE3DAttention(cfg)
    else:
        raise ValueError(
            f"Unsupported VAE config type in checkpoint: {type(cfg).__name__}"
        )
    state = clean_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, cfg, ckpt.get("stats_path")


def _save_volume_stats(
    stat_tensor: torch.Tensor,
    output_dir: Path,
    global_samples: int,
    tag: str,
    wandb_run,
) -> None:
    channels = stat_tensor.shape[0]
    mid = stat_tensor.shape[1] // 2
    fig, axes = plt.subplots(channels, 3, figsize=(12, 4 * channels), squeeze=False)
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
            ax.set_title(f"Ch {c} {tag}: {title}", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = output_dir / f"eval_{tag}_step_{global_samples}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    if wandb_run is not None:
        wandb.log({f"eval/{tag}": wandb.Image(fig)}, step=global_samples)
    plt.close(fig)


def _save_decoded_samples(
    decoded: torch.Tensor,
    output_dir: Path,
    global_samples: int,
    start_index: int,
    dtype: str,
) -> None:
    save_dir = output_dir / "decoded_samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    for i in range(decoded.shape[0]):
        idx = start_index + i
        out_path = save_dir / f"decoded_step_{global_samples}_sample_{idx}.npz"
        np.savez_compressed(
            out_path, volume=decoded[i].detach().cpu().numpy().astype(np_dtype)
        )


def _plot_latent_slices(
    latent: torch.Tensor,
    output_path: Path,
    z_indices: list[int],
) -> None:
    latent_np = latent.detach().cpu().float().numpy()
    num_channels = latent_np.shape[0]
    fig, axes = plt.subplots(
        num_channels,
        len(z_indices),
        figsize=(1.8 * len(z_indices), 2.0 * num_channels),
        squeeze=False,
    )

    for c in range(num_channels):
        for col, z in enumerate(z_indices):
            ax = axes[c, col]
            z_clamped = max(0, min(z, latent_np.shape[1] - 1))
            ax.imshow(latent_np[c, z_clamped], cmap="viridis")
            ax.set_title(f"ch{c} z={z_clamped}", fontsize=8)
            ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_decoded_slices(
    sample: torch.Tensor,
    output_path: Path,
    z_indices: list[int],
    channel_names: tuple[str, ...],
) -> None:
    sample_np = sample.detach().cpu().float().numpy()
    num_channels = sample_np.shape[0]
    fig, axes = plt.subplots(
        num_channels,
        len(z_indices),
        figsize=(1.8 * len(z_indices), 2.2 * num_channels),
        squeeze=False,
    )

    for c in range(num_channels):
        sample_c = np.nan_to_num(sample_np[c], nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(sample_c.min()) if sample_c.size else 0.0
        vmax = float(sample_c.max()) if sample_c.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        for col, z in enumerate(z_indices):
            ax = axes[c, col]
            ax.imshow(sample_c[z], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{channel_names[c]} z={z}", fontsize=8)
            ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_biflownet_training(args: argparse.Namespace, hooks: ObjectiveHooks) -> None:
    if args.decoded_dtype not in ("float16", "float32"):
        raise ValueError("decoded_dtype must be 'float16' or 'float32'.")
    if args.decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be > 0.")
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
    use_logvar = not args.no_logvar
    sample_filter = _build_latent_sample_filter(args)
    if accelerator.is_main_process and sample_filter is not None:
        logger.info(
            "Applying latent sample filter/weighter: %s", type(sample_filter).__name__
        )

    dataset = LatentZarrDataset(
        args.latents_zarr,
        pad_to=pad_to,
        use_logvar=use_logvar,
        sample_filter=sample_filter,
    )
    latent_channels, latent_spatial = _infer_latent_config(dataset)
    if accelerator.is_main_process:
        logger.info(
            "Latent shape (C, D, H, W): %s (channels=%d, spatial=%d)",
            dataset.latent_shape,
            latent_channels,
            latent_spatial,
        )

    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else output_dir / "latent_stats.json"
    )

    stats_mask = None
    if dataset.pad_amounts != (0, 0, 0):
        stats_mask = create_padding_mask(
            padded_shape=dataset.latent_shape,
            original_spatial=dataset.original_spatial,
            device=torch.device("cpu"),
        )

    if stats_path.exists() and not args.recompute_stats:
        if accelerator.is_main_process:
            logger.info("Loading latent stats from %s", stats_path)
    else:
        if accelerator.is_main_process:
            logger.info("Computing latent mean/std (distributed streaming)")
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
            save_latent_stats(stats_path, stats)
            logger.info("Saved latent stats to %s", stats_path)

    accelerator.wait_for_everyone()
    stats = load_latent_stats(stats_path)
    if stats["mean"].numel() != latent_channels:
        raise ValueError(
            f"Latent stats at {stats_path} have {stats['mean'].numel()} channel(s), "
            f"but model expects {latent_channels} (latent channels)."
        )

    latent_mean = stats["mean"].float().view(1, latent_channels, 1, 1, 1)
    if args.use_single_vae_scaling:
        global_std = get_global_std(stats, use_logvar)
        latent_std = global_std.view(1, 1, 1, 1, 1).expand(1, latent_channels, 1, 1, 1)
    else:
        latent_std = get_effective_std(stats, use_logvar).view(
            1, latent_channels, 1, 1, 1
        )

    dataset_size = len(dataset)
    holdout_size = max(1, math.ceil(dataset_size * 0.05))
    train_size = dataset_size - holdout_size
    holdout_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    if OVERFIT_SAMPLES:
        train_size = NUM_SAMPLES_TO_OVERFIT
    if train_size <= 0:
        raise ValueError(
            "Zarr store must contain more than 1 sample to reserve 1% for holdout."
        )
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))

    if accelerator.is_main_process and OVERFIT_SAMPLES:
        latent_plots_dir = output_dir / "latents"
        latent_plots_dir.mkdir(parents=True, exist_ok=True)
        for existing_plot in latent_plots_dir.glob("*_slices.png"):
            existing_plot.unlink()
        for sample_idx in range(len(train_dataset)):
            latent = train_dataset[sample_idx][1]
            if latent.ndim == 5:
                latent = latent[0]
            if latent.ndim != 4:
                raise ValueError(
                    f"Expected latent with shape [C, D, H, W], got {tuple(latent.shape)}"
                )
            depth = latent.shape[1]
            z_indices = [
                min(depth - 1, int(round(i * (depth - 1) / 3))) for i in range(4)
            ]
            _plot_latent_slices(
                latent, latent_plots_dir / f"{sample_idx}_slices.png", z_indices
            )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_kwargs)

    if len(args.dim_mults) != len(args.use_attn):
        raise ValueError(
            "dim_mults and use_attn must have the same length for BiFlowNet."
        )

    sub_volume_size = _to_3tuple(args.sub_volume_size)
    if any(latent_spatial % s != 0 for s in sub_volume_size):
        raise ValueError(
            f"Latent spatial {latent_spatial} not divisible by sub_volume_size {sub_volume_size}."
        )
    if args.vq_size != sub_volume_size[0] * 8:
        raise ValueError(
            f"Expected vq_size={sub_volume_size[0] * 8} for latent spatial {latent_spatial}, got {args.vq_size}."
        )

    raw_model = BiFlowNet(
        dim=args.model_dim,
        dim_mults=tuple(args.dim_mults),
        channels=latent_channels,
        init_kernel_size=args.init_kernel_size,
        cond_classes=None,
        learn_sigma=False,
        use_sparse_linear_attn=list(args.use_attn),
        vq_size=args.vq_size,
        num_mid_DiT=args.num_mid_dit,
        patch_size=args.patch_size,
        sub_volume_size=sub_volume_size,
        attn_heads=args.attn_heads,
        resnet_groups=args.resnet_groups,
        DiT_num_heads=args.dit_num_heads,
        mlp_ratio=args.mlp_ratio,
        res_condition=False,
    )
    model = hooks.wrap_model(raw_model, args)

    ema_model = EMAModel(model.parameters(), decay=0.9995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    total_optimizer_steps = args.max_steps // effective_batch_size
    warmup_steps = args.warmup_samples // effective_batch_size
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_optimizer_steps - warmup_steps), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    model, optimizer, train_loader, holdout_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, holdout_loader, scheduler
    )
    ema_model.to(accelerator.device)
    latent_mean = latent_mean.to(accelerator.device)
    latent_std = latent_std.to(accelerator.device)

    padding_mask = None
    if dataset.pad_amounts != (0, 0, 0):
        padding_mask = create_padding_mask(
            padded_shape=dataset.latent_shape,
            original_spatial=dataset.original_spatial,
            device=accelerator.device,
        )

    if args.vae_checkpoint is None:
        raise ValueError("vae_checkpoint must be set to decode samples.")
    vae_model, vae_cfg, vae_stats_path = _load_vae(
        Path(args.vae_checkpoint), torch.device("cpu")
    )
    if vae_cfg.latent_channels != latent_channels:
        raise ValueError(
            f"VAE latent_channels {vae_cfg.latent_channels} does not match dataset {latent_channels}."
        )
    data_stats_path = (
        Path(args.vae_stats_path)
        if args.vae_stats_path is not None
        else (Path(vae_stats_path) if vae_stats_path is not None else None)
    )
    if data_stats_path is None:
        raise ValueError("No VAE stats_path available for decoding.")
    data_stats = load_data_stats(data_stats_path)
    data_mean, data_std = reshape_stats_for_broadcast(
        data_stats, num_channels=vae_cfg.in_channels, num_dims=5
    )
    data_mean = data_mean.to("cpu")
    data_std = data_std.to("cpu")

    wandb_run = None
    if accelerator.is_main_process:
        init_kwargs = {
            "project": hooks.wandb_project,
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

    global_samples = 0
    log_loss_sum = 0.0
    log_weight_sum = 0.0
    next_log = args.log_every
    next_eval = args.eval_every_steps
    next_save = args.save_every
    last_saved_step = 0
    data_iter = iter(train_loader)
    progress_bar = (
        tqdm(total=args.max_steps, desc="Training", leave=True)
        if args.progress and accelerator.is_local_main_process
        else None
    )

    while global_samples < args.max_steps:
        try:
            batch_payload = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            continue

        model.train()
        with accelerator.accumulate(model):
            sample_weights, batch = batch_payload
            sample_weights = sample_weights.to(
                accelerator.device, non_blocking=True
            ).float()
            batch = batch.to(accelerator.device, non_blocking=True)
            batch = (batch - latent_mean) / latent_std
            if padding_mask is not None:
                batch = batch * padding_mask
            per_sample_loss = hooks.compute_per_sample_loss(
                model, batch, padding_mask, args
            )
            loss = (
                per_sample_loss * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1e-8)
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

        batch_size = batch.size(0)
        step_samples_int = int(
            accelerator.reduce(
                torch.tensor(batch_size, device=accelerator.device), reduction="sum"
            ).item()
        )
        global_samples += step_samples_int
        loss_num = accelerator.reduce(
            (per_sample_loss.detach() * sample_weights.detach()).sum(), reduction="sum"
        )
        loss_den = accelerator.reduce(sample_weights.detach().sum(), reduction="sum")
        log_loss_sum += loss_num.item()
        log_weight_sum += max(loss_den.item(), 1e-8)

        if progress_bar is not None:
            progress_bar.update(step_samples_int)

        if global_samples >= next_log:
            if accelerator.is_main_process and log_weight_sum > 0:
                logger.info(
                    "Samples %d | Train Loss %.6f",
                    global_samples,
                    log_loss_sum / log_weight_sum,
                )
            if wandb_run is not None and log_weight_sum > 0:
                wandb.log(
                    {
                        "train/loss": log_loss_sum / log_weight_sum,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_samples,
                )
            log_loss_sum = 0.0
            log_weight_sum = 0.0
            next_log += args.log_every

        if global_samples >= next_eval:
            model.eval()
            val_loss_sum = torch.tensor(0.0, device=accelerator.device)
            val_weight_sum = torch.tensor(0.0, device=accelerator.device)
            with torch.no_grad():
                for val_weights, val_batch in holdout_loader:
                    val_weights = val_weights.to(
                        accelerator.device, non_blocking=True
                    ).float()
                    val_batch = val_batch.to(accelerator.device, non_blocking=True)
                    val_batch = (val_batch - latent_mean) / latent_std
                    if padding_mask is not None:
                        val_batch = val_batch * padding_mask
                    with accelerator.autocast():
                        val_per_sample_loss = hooks.compute_per_sample_loss(
                            model, val_batch, padding_mask, args
                        )
                    val_loss_sum += (val_per_sample_loss.detach() * val_weights).sum()
                    val_weight_sum += val_weights.sum()

            val_loss_sum = accelerator.reduce(val_loss_sum, reduction="sum")
            val_weight_sum = accelerator.reduce(val_weight_sum, reduction="sum")
            if (
                accelerator.is_main_process
                and val_weight_sum.item() > 0
                and wandb_run is not None
            ):
                wandb.log(
                    {"val/loss": (val_loss_sum / val_weight_sum).item()},
                    step=global_samples,
                )

            total_eval = args.num_eval_samples
            num_procs = accelerator.num_processes
            rank = accelerator.process_index
            local_eval = total_eval // num_procs + (
                1 if rank < total_eval % num_procs else 0
            )

            decoded_sum = None
            decoded_sumsq = None
            decoded_count = torch.tensor(0, device=accelerator.device)
            saved = 0
            plot_sample = None

            if local_eval > 0:
                vae_model.to(accelerator.device)
                data_mean = data_mean.to(accelerator.device)
                data_std = data_std.to(accelerator.device)
                unwrapped_model = accelerator.unwrap_model(model)
                ema_model.store(unwrapped_model.parameters())
                ema_model.copy_to(unwrapped_model.parameters())
                samples = hooks.sample_latents(
                    unwrapped_model,
                    local_eval,
                    dataset.latent_shape,
                    accelerator.device,
                    padding_mask,
                    accelerator,
                    args,
                )
                ema_model.restore(unwrapped_model.parameters())
                samples = samples * latent_std + latent_mean

                if accelerator.is_main_process and samples.size(0) > 0:
                    first_sample = samples[0].detach().cpu().clone()
                    depth = first_sample.shape[1]
                    z_indices = [
                        min(depth - 1, int(round(i * (depth - 1) / 3)))
                        for i in range(4)
                    ]
                    first_latent_path = (
                        output_dir
                        / f"eval_first_generated_latent_step_{global_samples}.png"
                    )
                    _plot_latent_slices(first_sample, first_latent_path, z_indices)
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "eval/first_generated_latent": wandb.Image(
                                    first_latent_path
                                )
                            },
                            step=global_samples,
                        )

                for start in range(0, samples.shape[0], args.decode_batch_size):
                    chunk = samples[
                        start : min(start + args.decode_batch_size, samples.shape[0])
                    ]
                    with torch.no_grad():
                        with accelerator.autocast():
                            decoded = vae_model.decode(chunk)
                    decoded = _center_crop(
                        (decoded * data_std + data_mean).float(),
                        args.data_original_size,
                    )
                    if decoded_sum is None:
                        decoded_sum = torch.zeros_like(decoded[0])
                        decoded_sumsq = torch.zeros_like(decoded[0])
                    decoded_sum += decoded.sum(dim=0)
                    decoded_sumsq += (decoded**2).sum(dim=0)
                    decoded_count += decoded.size(0)
                    if (
                        accelerator.is_main_process
                        and plot_sample is None
                        and decoded.size(0) > 0
                    ):
                        plot_sample = decoded[0].detach().cpu().clone()
                    if (
                        accelerator.is_main_process
                        and args.save_decoded_samples
                        and saved < args.num_save_samples
                    ):
                        to_save = decoded[: args.num_save_samples - saved]
                        _save_decoded_samples(
                            to_save,
                            output_dir,
                            global_samples,
                            saved,
                            args.decoded_dtype,
                        )
                        saved += to_save.shape[0]

            if decoded_sum is None or decoded_sumsq is None:
                out_spatial = (
                    (args.data_original_size,) * 3
                    if args.data_original_size is not None
                    else _get_decoder_output_spatial(vae_cfg, dataset.latent_shape)
                )
                out_shape = (vae_cfg.in_channels, *out_spatial)
                decoded_sum = torch.zeros(
                    out_shape, device=accelerator.device, dtype=torch.float32
                )
                decoded_sumsq = torch.zeros_like(decoded_sum)

            decoded_sum = accelerator.reduce(decoded_sum, reduction="sum")
            decoded_sumsq = accelerator.reduce(decoded_sumsq, reduction="sum")
            decoded_count = accelerator.reduce(decoded_count, reduction="sum")
            if accelerator.is_main_process and decoded_count.item() > 0:
                mean_map = decoded_sum / decoded_count
                var_map = decoded_sumsq / decoded_count - mean_map**2
                std_map = var_map.clamp_min(0).sqrt()
                _save_volume_stats(
                    mean_map, output_dir, global_samples, "mean", wandb_run
                )
                _save_volume_stats(
                    std_map, output_dir, global_samples, "std", wandb_run
                )
                if plot_sample is not None:
                    if plot_sample.shape[0] >= 2:
                        plot_sample[1:2] = torch.pow(10.0, plot_sample[1:2]) - 1e-6
                    z_indices = list(
                        range(0, min(200, plot_sample.shape[1] - 1) + 1, 10)
                    )
                    channel_names = (
                        ("grav", "mag")
                        if plot_sample.shape[0] == 2
                        else tuple(f"ch{c}" for c in range(plot_sample.shape[0]))
                    )
                    plot_path = (
                        output_dir
                        / "decoded_plots"
                        / f"decoded_slices_step_{global_samples}.png"
                    )
                    _plot_decoded_slices(
                        plot_sample, plot_path, z_indices, channel_names
                    )
                    if wandb_run is not None:
                        wandb.log(
                            {"eval/decoded_slices": wandb.Image(plot_path)},
                            step=global_samples,
                        )

            vae_model.to("cpu")
            data_mean = data_mean.to("cpu")
            data_std = data_std.to("cpu")
            del decoded_sum, decoded_sumsq, decoded_count, plot_sample
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
                    "args": vars(args),
                    "latents_zarr": args.latents_zarr,
                    "original_spatial": dataset.original_spatial,
                    "latent_shape": dataset.latent_shape,
                    "latent_mean": latent_mean.cpu(),
                    "latent_std": latent_std.cpu(),
                    "use_single_vae_scaling": args.use_single_vae_scaling,
                    **hooks.checkpoint_extras(args),
                }
                checkpoint_path = (
                    output_dir / f"{hooks.checkpoint_prefix}_step_{global_samples}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
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
                "args": vars(args),
                "latents_zarr": args.latents_zarr,
                "original_spatial": dataset.original_spatial,
                "latent_shape": dataset.latent_shape,
                "latent_mean": latent_mean.cpu(),
                "latent_std": latent_std.cpu(),
                "use_single_vae_scaling": args.use_single_vae_scaling,
                **hooks.checkpoint_extras(args),
            }
            checkpoint_path = (
                output_dir / f"{hooks.checkpoint_prefix}_step_{global_samples}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
