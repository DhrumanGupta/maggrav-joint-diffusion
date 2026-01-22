"""
Evaluate a 3D VAE on the test split.

Usage:
    python src/evaluate_vae.py --zarr_path /path/to/zarr --checkpoint /path/to/ckpt.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..models.vae import VAE3D, VAE3DConfig
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


def _plot_recon_slices(
    sample: torch.Tensor,
    recon: torch.Tensor,
    output_path: Path,
    z_indices: list[int],
    channel_names: Tuple[str, str],
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
        vmin = float(min(sample_c.min(), recon_c.min()))
        vmax = float(max(sample_c.max(), recon_c.max()))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D VAE on test split")
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save sample/reconstruction/diff slices for z=0..200.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional cap on number of test samples to evaluate.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: Optional[int],
    accelerator: Accelerator,
) -> Dict[str, torch.Tensor]:
    model.eval()

    device = accelerator.device
    total_recon = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    ratio_sum = torch.zeros(2, dtype=torch.float32, device=device)

    eps = 1e-8
    progress = tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )
    for batch in progress:
        batch = batch.to(device, non_blocking=True)

        batch_norm = (batch - mean) / std
        with accelerator.autocast():
            recon, mu, logvar = model(batch_norm)
        recon_loss = F.mse_loss(recon, batch_norm)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        batch_size = batch.size(0)
        total_samples += batch_size
        total_recon += recon_loss.detach() * batch_size
        total_kl += kl_loss.detach() * batch_size

        recon_denorm = recon * std + mean
        batch_denorm = batch_norm * std + mean
        diff = recon_denorm - batch_denorm
        mse = diff.pow(2).mean(dim=(2, 3, 4))
        l2 = batch_denorm.pow(2).sum(dim=(2, 3, 4)).sqrt()
        ratio = mse / (l2 + eps)
        ratio_sum += ratio.sum(dim=0)

    total_recon = accelerator.reduce(total_recon, reduction="sum")
    total_kl = accelerator.reduce(total_kl, reduction="sum")
    total_samples = accelerator.reduce(total_samples, reduction="sum")
    ratio_sum = accelerator.reduce(ratio_sum, reduction="sum")

    if total_samples.item() == 0:
        return {
            "recon_loss": torch.tensor(0.0, device=device),
            "kl_loss": torch.tensor(0.0, device=device),
            "ratio_pct": torch.zeros(2, device=device),
        }

    avg_recon = total_recon / total_samples
    avg_kl = total_kl / total_samples
    ratio_avg = ratio_sum / total_samples
    ratio_pct = ratio_avg * 100.0
    return {"recon_loss": avg_recon, "kl_loss": avg_kl, "ratio_pct": ratio_pct}


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device, [VAE3DConfig])

    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else Path(checkpoint.get("stats_path", ""))
    )
    if not stats_path.exists():
        raise FileNotFoundError(
            "Stats file not found. Provide --stats_path or ensure checkpoint includes"
            f" stats_path. Got: {stats_path}"
        )

    stats = load_stats(stats_path)
    mean = stats["mean"].float().view(1, 2, 1, 1, 1).to(device)
    std = stats["std"].float().view(1, 2, 1, 1, 1).to(device)

    model_cfg = checkpoint["config"]
    model = VAE3D(model_cfg)
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

    metrics = evaluate(model, test_loader, mean, std, args.num_samples, accelerator)

    if accelerator.is_main_process:
        logger.info("Recon loss: %.6f", metrics["recon_loss"].item())
        logger.info("KL loss: %.6f", metrics["kl_loss"].item())
        ratio_pct = metrics["ratio_pct"]
        logger.info(
            "Per-channel MSE/L2 %%: channel0=%.4f%% | channel1=%.4f%%",
            ratio_pct[0].item(),
            ratio_pct[1].item(),
        )
        if args.plot:
            if len(test_subset) == 0:
                logger.info("No test samples available for plotting.")
            else:
                model.eval()
                sample = test_subset[0].unsqueeze(0).to(device)
                sample_norm = (sample - mean) / std
                with torch.no_grad():
                    with accelerator.autocast():
                        recon, _, _ = model(sample_norm)
                recon_denorm = recon * std + mean
                # Apply inverse log10 transformation to susceptibility channel (channel 1)
                recon_denorm[:, 1:2] = torch.pow(10.0, recon_denorm[:, 1:2]) - 1e-6
                depth = sample.shape[2]
                z_max = min(200, depth - 1)
                z_indices = list(range(0, z_max + 1, 10))
                plot_dir = checkpoint_path.parent / "eval_plots"
                plot_path = plot_dir / "vae_recon_slices.png"
                _plot_recon_slices(
                    sample[0],
                    recon_denorm[0],
                    plot_path,
                    z_indices,
                    channel_names=("grav", "mag"),
                )
                logger.info("Saved recon slices to %s", plot_path)


if __name__ == "__main__":
    main()
