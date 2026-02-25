"""
Evaluate a 3D VAE with attention on the test split.

Usage:
    python -m src.evaluate.evaluate_vae_attention --zarr_path /path/to/zarr --checkpoint /path/to/ckpt.pt
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..models.vae_attention import VAE3DAttention, VAE3DAttentionConfig
from ..utils.checkpoint import clean_state_dict, load_checkpoint
from ..utils.datasets import JointDensitySuscDataset
from ..utils.stats import load_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss on full 200³ volume."""
    return F.mse_loss(recon, target)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence loss on full latent."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Channel-wise normalize a batch."""
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of `normalize`."""
    return x * std + mean


def decode_for_taus(
    vae: torch.nn.Module,
    mu: torch.Tensor,
    taus: List[float],
    accelerator: Accelerator,
) -> Dict[float, torch.Tensor]:
    """Decode z = mu + eps * sqrt(tau) for a list of taus sharing the same eps."""
    device = mu.device
    eps = torch.randn_like(mu, device=device, dtype=mu.dtype)
    recons: Dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for tau in taus:
            with accelerator.autocast():
                z_tau = mu + eps * math.sqrt(tau)
                recons[tau] = vae.decode(z_tau)
    return recons


def compute_channel_plot_bounds(
    sample: torch.Tensor, recons: List[torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-channel vmin/vmax and diff bounds across sample and reconstructions."""
    sample_np = sample.detach().cpu().float().numpy()
    all_recon_np = np.stack(
        [r.detach().cpu().float().numpy() for r in recons],
        axis=0,
    )
    num_channels = sample_np.shape[0]
    vmin_per_ch = np.zeros(num_channels)
    vmax_per_ch = np.zeros(num_channels)
    diff_bound_per_ch = np.zeros(num_channels)

    for c in range(num_channels):
        sc = np.nan_to_num(sample_np[c], nan=0.0, posinf=0.0, neginf=0.0)
        rc_all = np.nan_to_num(all_recon_np[:, c], nan=0.0, posinf=0.0, neginf=0.0)
        vmin_per_ch[c] = float(min(sc.min(), rc_all.min()))
        vmax_per_ch[c] = float(max(sc.max(), rc_all.max()))
        diff_bound_per_ch[c] = 0.0
        for r in recons:
            rc = np.nan_to_num(
                r[c].detach().cpu().float().numpy(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            diff_c = rc - sc
            bound = np.max(np.abs(diff_c)) if diff_c.size else 0.0
            diff_bound_per_ch[c] = max(diff_bound_per_ch[c], float(bound))
        if diff_bound_per_ch[c] == 0.0:
            diff_bound_per_ch[c] = 1.0

    return vmin_per_ch, vmax_per_ch, diff_bound_per_ch


def _plot_recon_slices(
    sample: torch.Tensor,
    recon: torch.Tensor,
    output_path: Path,
    z_indices: list[int],
    channel_names: Tuple[str, str],
    vmin_per_ch: Optional[np.ndarray] = None,
    vmax_per_ch: Optional[np.ndarray] = None,
    diff_bound_per_ch: Optional[np.ndarray] = None,
) -> None:
    sample_np = sample.detach().cpu().float().numpy()
    recon_np = recon.detach().cpu().float().numpy()
    num_channels = sample_np.shape[0]
    num_cols = len(z_indices)
    num_rows = num_channels * 3
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(1.8 * num_cols, 2.2 * num_rows),
        squeeze=False,
    )

    for c in range(num_channels):
        sample_c = np.nan_to_num(sample_np[c], nan=0.0, posinf=0.0, neginf=0.0)
        recon_c = np.nan_to_num(recon_np[c], nan=0.0, posinf=0.0, neginf=0.0)
        diff_c = recon_c - sample_c
        if vmin_per_ch is not None and vmax_per_ch is not None:
            vmin = float(vmin_per_ch[c])
            vmax = float(vmax_per_ch[c])
        else:
            vmin = float(min(sample_c.min(), recon_c.min()))
            vmax = float(max(sample_c.max(), recon_c.max()))
        if diff_bound_per_ch is not None:
            diff_bound = float(diff_bound_per_ch[c])
        else:
            diff_bound = float(np.max(np.abs(diff_c))) if diff_c.size else 0.0
        if diff_bound == 0.0:
            diff_bound = 1.0

        row_base = c * 3
        for col, z in enumerate(z_indices):
            ax_sample = axes[row_base, col]
            ax_recon = axes[row_base + 1, col]
            ax_diff = axes[row_base + 2, col]

            ax_sample.imshow(sample_c[z], cmap="viridis", vmin=vmin, vmax=vmax)
            ax_recon.imshow(recon_c[z], cmap="viridis", vmin=vmin, vmax=vmax)
            ax_diff.imshow(
                diff_c[z], cmap="coolwarm", vmin=-diff_bound, vmax=diff_bound
            )

            ax_sample.set_title(f"{channel_names[c]} sample z={z}", fontsize=8)
            ax_recon.set_title(f"{channel_names[c]} recon z={z}", fontsize=8)
            ax_diff.set_title(f"{channel_names[c]} diff z={z}", fontsize=8)
            ax_sample.axis("off")
            ax_recon.axis("off")
            ax_diff.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_latent_slices(
    latent_mu: torch.Tensor,
    output_path: Path,
    z_indices: list[int],
) -> None:
    """Plot slices of the latent mean across all channels."""
    latent_np = latent_mu.detach().cpu().float().numpy()  # (C, D, H, W)
    num_channels = latent_np.shape[0]
    num_cols = len(z_indices)
    num_rows = num_channels

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(1.8 * num_cols, 2.0 * num_rows),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D VAE with attention on test split"
    )
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional cap on number of test samples to evaluate.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save sample/reconstruction/diff slices for z=0..200.",
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "List of taus for decoding z = mu + eps*tau (eps ~ N(0,1)). "
            "Example: --taus 0 0.5 1.0"
        ),
    )
    parser.add_argument(
        "--num_to_plot",
        type=int,
        default=10,
        help="Number of test samples to plot (when --plot). Each sample gets one set of tau plots.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    """Evaluate reconstruction and KL loss on the provided dataloader."""
    model.eval()
    device = accelerator.device

    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)

    progress = tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )
    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        batch = normalize(batch, mean, std)

        with accelerator.autocast():
            recon, mu, logvar = model(batch)
            recon_l = mse_loss(recon, batch)
            kl_l = kl_loss(mu, logvar)

        batch_size = batch.size(0)
        total_samples += batch_size
        total_recon += recon_l.detach() * batch_size
        total_kl += kl_l.detach() * batch_size

    total_recon = accelerator.reduce(total_recon, reduction="sum")
    total_kl = accelerator.reduce(total_kl, reduction="sum")
    total_samples = accelerator.reduce(total_samples, reduction="sum")

    if total_samples.item() == 0:
        return {
            "recon_loss": torch.tensor(0.0, device=device),
            "kl_loss": torch.tensor(0.0, device=device),
        }

    recon_avg = total_recon / total_samples
    kl_avg = total_kl / total_samples
    return {"recon_loss": recon_avg, "kl_loss": kl_avg}


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device, [VAE3DAttentionConfig])

    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else Path(checkpoint.get("stats_path", ""))
    )

    logger.info(f"Stats path: {stats_path}")

    if not stats_path.exists():
        raise FileNotFoundError(
            "Stats file not found. Provide --stats_path or ensure checkpoint includes"
            f" stats_path. Got: {stats_path}"
        )

    stats = load_stats(stats_path)
    mean = stats["mean"].float().view(1, 2, 1, 1, 1).to(device)
    std = stats["std"].float().view(1, 2, 1, 1, 1).to(device)

    model_cfg = checkpoint["config"]
    model = VAE3DAttention(model_cfg)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = JointDensitySuscDataset(args.zarr_path)
    dataset_size = len(dataset)
    train_size = int(0.98 * dataset_size)
    val_size = int(0.005 * dataset_size)
    test_subset = Subset(
        dataset,
        range(train_size + val_size, dataset_size),
    )

    if args.num_samples is not None:
        capped = min(len(test_subset), args.num_samples)
        test_subset = Subset(test_subset, range(0, capped))

    sampler = None
    if accelerator.num_processes > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_subset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
        if sampler is not None:
            sampler.set_epoch(0)

    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    if args.num_samples is not None:
        logger.info("Evaluating up to %d samples", args.num_samples)
    else:
        logger.info("Evaluating full test split (%d samples)", len(test_subset))

    metrics = evaluate(model, test_loader, mean, std, accelerator)

    if accelerator.is_main_process:
        logger.info("Recon loss: %.6f", metrics["recon_loss"].item())
        logger.info("KL loss: %.6f", metrics["kl_loss"].item())
        if args.plot:
            if len(test_subset) == 0:
                logger.info("No test samples available for plotting.")
            else:
                model.eval()
                vae = accelerator.unwrap_model(model)
                depth = dataset[0].shape[1]  # (C, D, H, W)
                z_max = min(200, depth - 1)
                z_indices = list(range(0, z_max + 1, 10))
                plot_dir = checkpoint_path.parent / "eval_plots"
                n_plot = min(args.num_to_plot, len(test_subset))

                for plot_idx in range(n_plot):
                    # Prepare per-sample directory
                    sample_dir = plot_dir / f"sample_{plot_idx}"
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    test_idx = train_size + val_size + plot_idx
                    sample = dataset[test_idx].unsqueeze(0).to(device)
                    sample_norm = normalize(sample, mean, std)
                    with torch.no_grad():
                        with accelerator.autocast():
                            mu, logvar = vae.encode(sample_norm)

                    # Plot latent mean slices (all channels) for this sample
                    latent_depth = mu.shape[2]
                    max_cols = 5
                    if latent_depth <= max_cols:
                        z_lat_indices = list(range(latent_depth))
                    else:
                        step = math.ceil(latent_depth / max_cols)
                        z_lat_indices = list(range(0, latent_depth, step))

                    # Save latent mean and stochastic samples in a 'latents' subfolder
                    latents_dir = sample_dir / "latents"
                    latents_dir.mkdir(parents=True, exist_ok=True)

                    # Raw latent mean (mu)
                    raw_path = latents_dir / "raw.png"
                    _plot_latent_slices(
                        mu[0],
                        raw_path,
                        z_lat_indices,
                    )

                    # Five stochastic latent samples: mu + exp(0.5 * logvar) * eps
                    latent_std = torch.exp(0.5 * logvar)
                    for i in range(5):
                        eps = torch.randn_like(mu)
                        z_sample = mu + latent_std * eps
                        latent_sample_path = latents_dir / f"latent_{i}.png"
                        _plot_latent_slices(
                            z_sample[0],
                            latent_sample_path,
                            z_lat_indices,
                        )

                    sample_plot = sample.clone()
                    sample_plot[:, 1:2] = torch.pow(10.0, sample_plot[:, 1:2]) - 1e-6

                    # Build all recons for all taus (same sample, same eps across taus)
                    torch.manual_seed(42 + plot_idx)
                    recons_per_tau = decode_for_taus(vae, mu, args.taus, accelerator)

                    recons_denorm_list = []
                    for tau in args.taus:
                        recon = recons_per_tau[tau]
                        recon_denorm = denormalize(recon, mean, std)
                        recon_denorm[:, 1:2] = (
                            torch.pow(10.0, recon_denorm[:, 1:2]) - 1e-6
                        )
                        recons_denorm_list.append(recon_denorm[0])

                    # Global color scale: same vmin/vmax/diff_bound across sample and all taus
                    vmin_per_ch, vmax_per_ch, diff_bound_per_ch = (
                        compute_channel_plot_bounds(sample_plot[0], recons_denorm_list)
                    )

                    for tau, recon_denorm_0 in zip(args.taus, recons_denorm_list):
                        plot_path = (
                            sample_dir / f"vae_attention_recon_slices_tau_{tau}.png"
                        )
                        _plot_recon_slices(
                            sample_plot[0],
                            recon_denorm_0,
                            plot_path,
                            z_indices,
                            channel_names=("grav", "mag"),
                            vmin_per_ch=vmin_per_ch,
                            vmax_per_ch=vmax_per_ch,
                            diff_bound_per_ch=diff_bound_per_ch,
                        )
                        logger.info("Saved recon slices to %s", plot_path)


if __name__ == "__main__":
    main()
