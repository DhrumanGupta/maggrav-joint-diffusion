"""
Train a 3D VAE with attention on density + susceptibility volumes.

Usage:
    python -m src.train.train_vae_attention --config config/train_vae_attention.yaml
    accelerate launch -m src.train.train_vae_attention --config config/train_vae_attention.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import zarr
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

import wandb

from ..data.filtering import (load_filter_stats_cache,
                              select_indices_from_filter_stats)
from ..data.stats_utils import (compute_streaming_stats_ddp, load_stats,
                                save_stats)
from ..models.vae_attention import VAE3DAttention, VAE3DAttentionConfig
from ..utils.datasets import JointDensitySuscDataset

INPUT_SIZE = 200
LATENT_SIZE = 16

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss on full volume."""
    return F.mse_loss(recon, target)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence loss on full latent."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def build_filtered_train_indices(
    zarr_path: str,
    train_size: int,
    susc_active_threshold_log10: float,
    min_active_frac: float,
    low_info_keep_prob: float,
    filter_batch_size: int,
    filter_num_workers: int,
    filter_prefetch_factor: int,
    filter_persistent_workers: bool,
    seed: int,
    show_progress: bool,
    cache_path: Path,
    recompute_cache: bool,
    accelerator: Accelerator,
) -> Dict[str, object]:
    """Select train indices by downsampling low-information susceptibility samples."""
    if train_size <= 0:
        return {
            "selected_indices": [],
            "original_count": 0,
            "informative_count": 0,
            "low_info_count": 0,
            "retained_low_info_count": 0,
        }

    cache_format_version = 3
    max_rock_id = 255

    class _RockCountFilterDataset(Dataset):
        """Filter-only dataset: returns rock ids and per-sample susceptibility LUT."""

        def __init__(self, zarr_path_: str):
            self.zarr_path = zarr_path_
            self._group = None
            self._rock_types = None
            self._meta_array = None
            self._meta_attr = None
            self._meta_list = None

        def __getstate__(self):
            state = self.__dict__.copy()
            state["_group"] = None
            state["_rock_types"] = None
            state["_meta_array"] = None
            return state

        def _ensure_open(self) -> None:
            if self._group is not None:
                return
            group = zarr.open_group(self.zarr_path, mode="r")
            if "rock_types" not in group:
                raise ValueError("Zarr store missing 'rock_types' array.")
            self._group = group
            self._rock_types = group["rock_types"]
            if "samples_metadata" in group:
                self._meta_array = group["samples_metadata"]
            else:
                self._meta_array = None
                self._meta_attr = group.attrs.get("samples_metadata")
                self._meta_list = None
            if self._meta_array is None and not self._meta_attr:
                raise ValueError("samples_metadata not found in Zarr store.")

        def __len__(self) -> int:
            self._ensure_open()
            return int(self._rock_types.shape[0])

        def _load_metadata(self, idx: int) -> dict:
            if self._meta_array is not None:
                meta_raw = self._meta_array[idx]
            elif self._meta_attr:
                if self._meta_list is None:
                    parsed = json.loads(self._meta_attr)
                    if not isinstance(parsed, list):
                        raise ValueError("samples_metadata attribute is not a list.")
                    self._meta_list = parsed
                meta_raw = self._meta_list[idx]
            else:
                raise ValueError("samples_metadata not available.")

            if isinstance(meta_raw, np.generic):
                meta_raw = meta_raw.item()
            if isinstance(meta_raw, np.ndarray):
                if meta_raw.shape == ():
                    meta_raw = meta_raw.item()
                else:
                    raise ValueError(
                        f"Unexpected metadata array at index {idx}: shape={meta_raw.shape}"
                    )
            if isinstance(meta_raw, str):
                meta_raw = json.loads(meta_raw)
            if not isinstance(meta_raw, dict):
                raise ValueError(
                    f"Unexpected metadata format at index {idx}: {meta_raw!r}"
                )
            return meta_raw

        def _build_susc_log10_lut(self, meta: dict) -> torch.Tensor:
            lut = torch.full((max_rock_id + 1,), float("-inf"), dtype=torch.float32)
            properties = meta.get("properties", {})
            if not isinstance(properties, dict):
                return lut
            for key, values in properties.items():
                try:
                    rid = int(key)
                    if rid < 0 or rid > max_rock_id:
                        continue
                    susc_val = float(values[1])
                    susc_log10 = torch.log10(
                        torch.tensor(susc_val + 1.0e-6, dtype=torch.float32)
                    ).item()
                    lut[rid] = float(susc_log10)
                except (TypeError, ValueError, IndexError):
                    continue
            return lut

        def __getitem__(self, idx: int):
            self._ensure_open()
            rock_types = torch.as_tensor(self._rock_types[idx], dtype=torch.int16)
            meta = self._load_metadata(idx)
            susc_log10_lut = self._build_susc_log10_lut(meta)
            return idx, rock_types, susc_log10_lut

    def load_filter_stats(cache_file: Path) -> Optional[Dict[str, object]]:
        if not cache_file.exists() or recompute_cache:
            return None
        payload = load_filter_stats_cache(
            cache_path=cache_file,
            expected_train_size=train_size,
            expected_format=cache_format_version,
            strict=False,
        )
        if payload is None and accelerator.is_main_process:
            logger.info(
                "Ignoring filter cache because it is missing or incompatible: %s",
                cache_file,
            )
        return payload

    def compute_filter_stats(cache_file: Path) -> Dict[str, object]:
        dataset = _RockCountFilterDataset(zarr_path)
        rank = int(accelerator.process_index)
        world_size = int(accelerator.num_processes)
        local_candidate_indices = list(range(rank, train_size, world_size))
        candidate_subset = Subset(dataset, local_candidate_indices)
        filter_loader_kwargs = {
            "batch_size": max(1, int(filter_batch_size)),
            "shuffle": False,
            "num_workers": filter_num_workers,
            "pin_memory": True,
        }
        if filter_num_workers > 0:
            filter_loader_kwargs["prefetch_factor"] = int(filter_prefetch_factor)
            filter_loader_kwargs["persistent_workers"] = bool(filter_persistent_workers)
        loader = DataLoader(candidate_subset, **filter_loader_kwargs)

        device = accelerator.device
        if accelerator.is_main_process:
            logger.info(
                "Computing rock-count filter cache across %d rank(s) on %s (batch=%d workers=%d prefetch=%d persistent=%s)",
                world_size,
                device.type,
                int(filter_batch_size),
                int(filter_num_workers),
                int(filter_prefetch_factor),
                bool(filter_persistent_workers),
            )

        all_indices: List[torch.Tensor] = []
        all_rock_id_counts: List[torch.Tensor] = []
        all_susc_log10_lut: List[torch.Tensor] = []
        voxel_count = None

        progress = tqdm(
            loader,
            desc="Filtering train samples",
            disable=not show_progress,
        )
        local_total_samples = len(local_candidate_indices)
        processed_local_samples = 0
        filter_start_time = time.perf_counter()
        last_log_time = filter_start_time
        last_log_samples = 0
        for step_idx, (
            batch_indices,
            rock_types_batch,
            susc_log10_lut_batch,
        ) in enumerate(progress, start=1):
            rock_types_batch = rock_types_batch.to(dtype=torch.int64)

            if voxel_count is None:
                voxel_count = int(rock_types_batch[0].numel())

            rock_ids = rock_types_batch.view(rock_types_batch.shape[0], -1)
            rock_ids = rock_ids.clamp(min=0, max=max_rock_id)
            batch_n = int(rock_ids.shape[0])
            processed_local_samples += batch_n
            offsets = (
                torch.arange(batch_n, device=rock_ids.device, dtype=torch.int64)
                .unsqueeze(1)
                .mul(max_rock_id + 1)
            )
            flat_with_offsets = (rock_ids + offsets).reshape(-1)
            flat_counts = torch.bincount(
                flat_with_offsets, minlength=batch_n * (max_rock_id + 1)
            )
            batch_rock_counts = flat_counts.view(batch_n, max_rock_id + 1)

            all_indices.append(batch_indices.to(dtype=torch.int64).cpu())
            all_rock_id_counts.append(batch_rock_counts.to(dtype=torch.int64).cpu())
            all_susc_log10_lut.append(
                susc_log10_lut_batch.to(dtype=torch.float32).cpu()
            )

            if accelerator.is_local_main_process and (step_idx % 50 == 0):
                now = time.perf_counter()
                elapsed = now - filter_start_time
                window_elapsed = now - last_log_time
                avg_rate = (
                    float(processed_local_samples) / elapsed if elapsed > 0.0 else 0.0
                )
                window_rate = (
                    float(processed_local_samples - last_log_samples) / window_elapsed
                    if window_elapsed > 0.0
                    else 0.0
                )
                remaining = max(0, local_total_samples - processed_local_samples)
                eta_seconds = remaining / avg_rate if avg_rate > 0.0 else float("inf")
                eta_minutes = (
                    eta_seconds / 60.0 if eta_seconds != float("inf") else float("inf")
                )
                logger.info(
                    "Filter cache progress (local rank): %d/%d samples | avg %.1f samp/s | window %.1f samp/s | ETA %.1f min",
                    processed_local_samples,
                    local_total_samples,
                    avg_rate,
                    window_rate,
                    eta_minutes,
                )
                last_log_time = now
                last_log_samples = processed_local_samples

        if voxel_count is None:
            voxel_count = INPUT_SIZE * INPUT_SIZE * INPUT_SIZE

        if accelerator.is_local_main_process:
            total_elapsed = time.perf_counter() - filter_start_time
            overall_rate = (
                float(processed_local_samples) / total_elapsed
                if total_elapsed > 0.0
                else 0.0
            )
            logger.info(
                "Filter cache local scan complete: %d samples in %.1fs (%.1f samp/s)",
                processed_local_samples,
                total_elapsed,
                overall_rate,
            )

        local_sample_indices = (
            torch.cat(all_indices, dim=0)
            if all_indices
            else torch.empty(0, dtype=torch.int64)
        )
        local_rock_id_counts = (
            torch.cat(all_rock_id_counts, dim=0)
            if all_rock_id_counts
            else torch.empty((0, max_rock_id + 1), dtype=torch.int64)
        )
        local_susc_log10_lut = (
            torch.cat(all_susc_log10_lut, dim=0)
            if all_susc_log10_lut
            else torch.empty((0, max_rock_id + 1), dtype=torch.float32)
        )
        if accelerator.num_processes > 1:
            local_count = torch.tensor(
                [int(local_sample_indices.shape[0])],
                device=device,
                dtype=torch.int64,
            )
            gathered_counts = [
                torch.zeros_like(local_count) for _ in range(accelerator.num_processes)
            ]
            dist.all_gather(gathered_counts, local_count)
            counts_per_rank = [int(t.item()) for t in gathered_counts]
            max_count = max(counts_per_rank) if counts_per_rank else 0

            send_indices = torch.full(
                (max_count,),
                fill_value=-1,
                device=device,
                dtype=torch.int64,
            )
            send_hist = torch.zeros(
                (max_count, max_rock_id + 1),
                device=device,
                dtype=torch.int64,
            )
            send_lut = torch.full(
                (max_count, max_rock_id + 1),
                float("-inf"),
                device=device,
                dtype=torch.float32,
            )
            local_count_int = int(local_count.item())
            if local_count_int > 0:
                send_indices[:local_count_int] = local_sample_indices.to(device=device)
                send_hist[:local_count_int] = local_rock_id_counts.to(device=device)
                send_lut[:local_count_int] = local_susc_log10_lut.to(device=device)

            gathered_indices = [
                torch.empty_like(send_indices) for _ in range(world_size)
            ]
            gathered_hists = [torch.empty_like(send_hist) for _ in range(world_size)]
            gathered_luts = [torch.empty_like(send_lut) for _ in range(world_size)]
            dist.all_gather(gathered_indices, send_indices)
            dist.all_gather(gathered_hists, send_hist)
            dist.all_gather(gathered_luts, send_lut)

            if accelerator.is_main_process:
                merged_indices_parts: List[torch.Tensor] = []
                merged_hist_parts: List[torch.Tensor] = []
                merged_lut_parts: List[torch.Tensor] = []
                for rank_idx, rank_count in enumerate(counts_per_rank):
                    if rank_count <= 0:
                        continue
                    merged_indices_parts.append(
                        gathered_indices[rank_idx][:rank_count].cpu()
                    )
                    merged_hist_parts.append(
                        gathered_hists[rank_idx][:rank_count].cpu()
                    )
                    merged_lut_parts.append(gathered_luts[rank_idx][:rank_count].cpu())
                merged_sample_indices = (
                    torch.cat(merged_indices_parts, dim=0)
                    if merged_indices_parts
                    else torch.empty(0, dtype=torch.int64)
                )
                merged_rock_id_counts = (
                    torch.cat(merged_hist_parts, dim=0)
                    if merged_hist_parts
                    else torch.empty((0, max_rock_id + 1), dtype=torch.int64)
                )
                merged_susc_log10_lut = (
                    torch.cat(merged_lut_parts, dim=0)
                    if merged_lut_parts
                    else torch.empty((0, max_rock_id + 1), dtype=torch.float32)
                )
            else:
                merged_sample_indices = torch.empty(0, dtype=torch.int64)
                merged_rock_id_counts = torch.empty(
                    (0, max_rock_id + 1), dtype=torch.int64
                )
                merged_susc_log10_lut = torch.empty(
                    (0, max_rock_id + 1), dtype=torch.float32
                )
        else:
            merged_sample_indices = local_sample_indices
            merged_rock_id_counts = local_rock_id_counts
            merged_susc_log10_lut = local_susc_log10_lut

        if accelerator.is_main_process:
            merged_payload = {
                "cache_format_version": cache_format_version,
                "filter_backend": "rock_counts",
                "train_size": int(train_size),
                "sample_indices": merged_sample_indices,
                "rock_id_counts": merged_rock_id_counts,
                "susc_log10_lut": merged_susc_log10_lut,
                "voxel_count_per_sample": int(voxel_count),
            }
            sort_order = torch.argsort(merged_payload["sample_indices"])
            merged_payload["sample_indices"] = merged_payload["sample_indices"][
                sort_order
            ]
            merged_payload["rock_id_counts"] = merged_payload["rock_id_counts"][
                sort_order
            ]
            merged_payload["susc_log10_lut"] = merged_payload["susc_log10_lut"][
                sort_order
            ]
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(merged_payload, cache_file)
            logger.info(
                "Saved rock-count filter cache statistics: %s",
                cache_file,
            )
        accelerator.wait_for_everyone()
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    filter_stats = load_filter_stats(cache_path)
    if filter_stats is None:
        filter_stats = compute_filter_stats(cache_path)
    else:
        if accelerator.is_main_process:
            logger.info("Loaded filter cache statistics: %s", cache_path)

    try:
        selection = select_indices_from_filter_stats(
            filter_stats=filter_stats,
            susc_active_threshold_log10=float(susc_active_threshold_log10),
            min_active_frac=float(min_active_frac),
            low_info_keep_prob=float(low_info_keep_prob),
            seed=int(seed),
        )
    except KeyError as exc:
        raise ValueError(
            "Filter cache payload missing rock-count fields; "
            "set filter_cache_recompute=true once to rebuild filter cache."
        ) from exc
    selected_indices = selection["selected_indices"]
    informative_count = int(selection["informative_count"])
    low_info_count = int(selection["low_info_count"])
    retained_low_info_count = int(selection["retained_low_info_count"])

    logger.info(
        "Filtering complete: kept %d / %d train samples",
        len(selected_indices),
        train_size,
    )
    logger.info(
        "Rock-count filter cache includes %d samples",
        int(selection["cache_sample_count"]),
    )

    return {
        "selected_indices": selected_indices,
        "original_count": train_size,
        "informative_count": informative_count,
        "low_info_count": low_info_count,
        "retained_low_info_count": retained_low_info_count,
        "cache_path": str(cache_path),
    }


def load_selected_train_indices(
    selected_indices_path: Path,
    expected_train_size: int,
) -> Optional[List[int]]:
    """Load cached selected train indices from disk."""
    if not selected_indices_path.exists():
        return None

    with selected_indices_path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Selected indices payload must be a dict: {selected_indices_path}"
        )

    selected = payload.get("selected_indices")
    if not isinstance(selected, list):
        raise ValueError(
            f"Missing/invalid 'selected_indices' in {selected_indices_path}"
        )

    cached_train_size = payload.get("train_size")
    if cached_train_size is not None and int(cached_train_size) != int(
        expected_train_size
    ):
        logger.warning(
            "Ignoring selected indices cache because train_size mismatched "
            "(cache=%s current=%s): %s",
            cached_train_size,
            expected_train_size,
            selected_indices_path,
        )
        return None

    indices = [int(i) for i in selected]
    if any(i < 0 or i >= expected_train_size for i in indices):
        raise ValueError(
            f"Selected indices out of range [0, {expected_train_size}): "
            f"{selected_indices_path}"
        )
    return indices


def save_selected_train_indices(
    selected_indices_path: Path,
    selected_indices: List[int],
    train_size: int,
    seed: int,
) -> None:
    """Save selected train indices for deterministic reuse."""
    selected_indices_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_size": int(train_size),
        "seed": int(seed),
        "selected_count": int(len(selected_indices)),
        "selected_indices": [int(i) for i in selected_indices],
    }
    with selected_indices_path.open("w") as f:
        json.dump(payload, f, indent=2)


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
        batch = (batch - mean) / std

        with accelerator.autocast():
            recon, mu, logvar = model(batch)
            recon_loss = mse_loss(recon, batch)
            kl = kl_loss(mu, logvar)
            loss = recon_loss + kl_weight * kl

        batch_size = batch.size(0)
        total_samples += batch_size
        total_recon += recon_loss.detach() * batch_size
        total_kl += kl.detach() * batch_size
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
    parser = argparse.ArgumentParser(description="Train 3D VAE with Attention")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_vae_attention.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--wandb_resume_id",
        type=str,
        default=None,
        help="Optional W&B run id to resume. If not provided, uses id from checkpoint when available.",
    )
    parser.add_argument(
        "--kl_weight", type=float, default=None, help="Override kl_weight from config"
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=None,
        help="Override latent_channels from config",
    )
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping/dict: {cli_args.config}")

    config.setdefault("resume_checkpoint", None)
    config.setdefault("wandb_resume_id", None)

    # Apply CLI overrides
    if cli_args.kl_weight is not None:
        config["kl_weight"] = cli_args.kl_weight
    if cli_args.latent_channels is not None:
        config["latent_channels"] = cli_args.latent_channels
    if cli_args.resume_checkpoint is not None:
        config["resume_checkpoint"] = cli_args.resume_checkpoint
    if cli_args.wandb_resume_id is not None:
        config["wandb_resume_id"] = cli_args.wandb_resume_id

    return argparse.Namespace(**config)


def main() -> None:
    args = parse_args()
    if not hasattr(args, "kl_warmup_steps"):
        args.kl_warmup_steps = 0
    if not hasattr(args, "warmup_steps"):
        args.warmup_steps = 0
    if not hasattr(args, "kl_zero_steps"):
        args.kl_zero_steps = 0
    args.kl_zero_steps = int(args.kl_zero_steps)
    if args.kl_zero_steps < 0:
        raise ValueError("kl_zero_steps must be >= 0")
    if not hasattr(args, "susc_active_threshold_log10"):
        args.susc_active_threshold_log10 = -2.5
    if not hasattr(args, "min_active_frac"):
        args.min_active_frac = 0.005
    if not hasattr(args, "low_info_keep_prob"):
        args.low_info_keep_prob = 0.2
    if not hasattr(args, "filter_cache_path"):
        args.filter_cache_path = None
    if not hasattr(args, "selected_indices_path"):
        args.selected_indices_path = None
    if not hasattr(args, "filter_cache_recompute"):
        args.filter_cache_recompute = False
    if not hasattr(args, "prefetch_factor"):
        args.prefetch_factor = 4
    if not hasattr(args, "persistent_workers"):
        args.persistent_workers = True
    if not hasattr(args, "filter_batch_size"):
        args.filter_batch_size = 64
    if not hasattr(args, "filter_num_workers"):
        args.filter_num_workers = 12
    if not hasattr(args, "filter_prefetch_factor"):
        args.filter_prefetch_factor = 8
    if not hasattr(args, "filter_persistent_workers"):
        args.filter_persistent_workers = True

    args.susc_active_threshold_log10 = float(args.susc_active_threshold_log10)
    args.min_active_frac = float(args.min_active_frac)
    args.low_info_keep_prob = float(args.low_info_keep_prob)
    args.prefetch_factor = int(args.prefetch_factor)
    args.persistent_workers = bool(args.persistent_workers)
    args.filter_batch_size = int(args.filter_batch_size)
    args.filter_num_workers = int(args.filter_num_workers)
    args.filter_prefetch_factor = int(args.filter_prefetch_factor)
    args.filter_persistent_workers = bool(args.filter_persistent_workers)
    if args.min_active_frac < 0.0 or args.min_active_frac > 1.0:
        raise ValueError("min_active_frac must be in [0, 1]")
    if args.low_info_keep_prob < 0.0 or args.low_info_keep_prob > 1.0:
        raise ValueError("low_info_keep_prob must be in [0, 1]")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")
    if args.filter_batch_size <= 0:
        raise ValueError("filter_batch_size must be > 0")
    if args.filter_num_workers < 0:
        raise ValueError("filter_num_workers must be >= 0")
    if args.filter_prefetch_factor <= 0:
        raise ValueError("filter_prefetch_factor must be > 0")

    # Initialize CUDA context BEFORE creating Accelerator to avoid NCCL errors
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.cuda.init()
        torch.cuda.synchronize()
        _ = torch.empty(1, device=f"cuda:{local_rank}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    set_seed(args.seed)

    output_dir = (
        Path(args.output_dir)
        / f"latent_channels_{args.latent_channels}"
        / f"kl_weight_{args.kl_weight}"
    )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else output_dir / "vae_stats.json"
    )

    dataset = JointDensitySuscDataset(args.zarr_path)
    dataset_size = len(dataset)
    train_size = int(0.98 * dataset_size)
    val_size = int(0.005 * dataset_size)
    test_size = dataset_size - train_size - val_size

    filter_cache_path = (
        Path(args.filter_cache_path)
        if args.filter_cache_path
        else output_dir / "train_filter_stats.pt"
    )
    selected_indices_path = (
        Path(args.selected_indices_path)
        if args.selected_indices_path
        else output_dir / "train_selected_indices.json"
    )
    loaded_selected_indices = False
    selected_train_indices = None
    if not bool(args.filter_cache_recompute):
        selected_train_indices = load_selected_train_indices(
            selected_indices_path=selected_indices_path,
            expected_train_size=train_size,
        )
        loaded_selected_indices = selected_train_indices is not None
        if loaded_selected_indices and accelerator.is_main_process:
            logger.info(
                "Loaded selected train indices: %s (count=%d)",
                selected_indices_path,
                len(selected_train_indices),
            )

    filter_info = {
        "selected_indices": [],
        "original_count": train_size,
        "informative_count": 0,
        "low_info_count": 0,
        "retained_low_info_count": 0,
        "cache_path": str(filter_cache_path),
    }
    if selected_train_indices is None:
        filter_info = build_filtered_train_indices(
            zarr_path=args.zarr_path,
            train_size=train_size,
            susc_active_threshold_log10=args.susc_active_threshold_log10,
            min_active_frac=args.min_active_frac,
            low_info_keep_prob=args.low_info_keep_prob,
            filter_batch_size=args.filter_batch_size,
            filter_num_workers=args.filter_num_workers,
            filter_prefetch_factor=args.filter_prefetch_factor,
            filter_persistent_workers=args.filter_persistent_workers,
            seed=args.seed,
            show_progress=accelerator.is_local_main_process,
            cache_path=filter_cache_path,
            recompute_cache=bool(args.filter_cache_recompute),
            accelerator=accelerator,
        )
        selected_train_indices = filter_info["selected_indices"]
        if not selected_train_indices:
            if accelerator.is_main_process:
                logger.warning(
                    "Filtering selected 0 train samples; falling back to original train split."
                )
            selected_train_indices = list(range(0, train_size))
        if accelerator.is_main_process:
            save_selected_train_indices(
                selected_indices_path=selected_indices_path,
                selected_indices=selected_train_indices,
                train_size=train_size,
                seed=args.seed,
            )
            logger.info("Saved selected train indices: %s", selected_indices_path)

    accelerator.wait_for_everyone()
    selected_train_indices = load_selected_train_indices(
        selected_indices_path=selected_indices_path,
        expected_train_size=train_size,
    )
    if selected_train_indices is None:
        raise RuntimeError(
            "Selected train indices were not found after filtering. "
            f"Expected: {selected_indices_path}"
        )

    if stats_path.exists() and not args.recompute_stats:
        if accelerator.is_main_process:
            logger.info(f"Loading stats from {stats_path}")
    else:
        logger.info(
            "Computing global mean/std (distributed streaming) on filtered train subset"
        )
        stats_dataset = Subset(dataset, selected_train_indices)
        stats = compute_streaming_stats_ddp(
            stats_dataset,
            args.stats_batch_size,
            args.num_workers,
            accelerator,
            num_channels=2,
        )
        if accelerator.is_main_process:
            save_stats(stats_path, stats)
            logger.info(f"Saved stats to {stats_path}")

    accelerator.wait_for_everyone()
    stats = load_stats(stats_path)

    mean = stats["mean"].float().view(1, 2, 1, 1, 1)
    std = stats["std"].float().view(1, 2, 1, 1, 1)

    train_subset = Subset(dataset, selected_train_indices)
    val_subset = Subset(dataset, range(train_size, train_size + val_size))
    test_subset = Subset(
        dataset,
        range(train_size + val_size, dataset_size),
    )

    loader_common_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_common_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_common_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )

    if accelerator.is_main_process:
        original_count = int(filter_info["original_count"])
        informative_count = int(filter_info["informative_count"])
        low_info_count = int(filter_info["low_info_count"])
        retained_low_info_count = int(filter_info["retained_low_info_count"])
        final_train_count = len(selected_train_indices)
        retention_ratio = (
            float(final_train_count) / float(original_count)
            if original_count > 0
            else 0.0
        )
        logger.info(
            "Train filtering | threshold=%.3f min_active_frac=%.5f low_info_keep_prob=%.3f",
            args.susc_active_threshold_log10,
            args.min_active_frac,
            args.low_info_keep_prob,
        )
        logger.info(
            "Train filtering counts | original=%d informative=%d low_info=%d retained_low_info=%d final=%d (retention=%.2f%%)",
            original_count,
            informative_count,
            low_info_count,
            retained_low_info_count,
            final_train_count,
            retention_ratio * 100.0,
        )
        logger.info("Train filter cache path: %s", filter_info["cache_path"])
        logger.info("Train selected indices path: %s", selected_indices_path)
        logger.info(
            "Train selected indices source: %s",
            "cached selection" if loaded_selected_indices else "freshly filtered",
        )
        logger.info(
            "Dataset split sizes - train: %d | val: %d | test: %d",
            len(selected_train_indices),
            val_size,
            test_size,
        )
    accelerator.wait_for_everyone()

    # Build model config from args
    model_cfg = VAE3DAttentionConfig(
        in_channels=2,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        bottleneck_channels=getattr(args, "bottleneck_channels", 192),
        latent_size=LATENT_SIZE,
        blocks_per_stage=args.blocks_per_stage,
        num_attention_blocks=getattr(args, "num_attention_blocks", 2),
    )
    model = VAE3DAttention(model_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    resume_checkpoint = args.resume_checkpoint
    resume_data = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if accelerator.is_main_process:
            logger.info("Resuming from checkpoint: %s", resume_path)
        resume_data = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(resume_data["model_state_dict"])
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])

    if accelerator.is_main_process:
        logger.info(
            f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)

    wandb_run = None
    if accelerator.is_main_process:
        resume_id = None
        if args.wandb_resume_id:
            resume_id = args.wandb_resume_id
        elif isinstance(resume_data, dict):
            resume_id = resume_data.get("wandb_run_id")
        init_kwargs = {
            "project": os.getenv("WANDB_PROJECT", "maggrav-vae-attention"),
            "config": {
                **vars(args),
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
            },
        }
        if resume_id:
            init_kwargs["id"] = resume_id
            init_kwargs["resume"] = "allow"
        run_name = os.getenv("WANDB_RUN_NAME")
        if run_name:
            init_kwargs["name"] = run_name
        wandb_run = wandb.init(**init_kwargs)

    if accelerator.is_main_process:
        logger.info("Training for %d samples", args.max_steps)

    global_samples = 0
    if isinstance(resume_data, dict) and "global_samples" in resume_data:
        global_samples = int(resume_data["global_samples"])
    best_val_loss = float("inf")
    best_path = output_dir / "vae_attention_best.json"
    log_samples = 0
    log_total_sum = 0.0
    log_recon_sum = 0.0
    log_kl_sum = 0.0
    eval_samples = 0
    eval_total_sum = 0.0
    eval_recon_sum = 0.0
    eval_kl_sum = 0.0

    def next_multiple(step: int, interval: int) -> int:
        if interval <= 0:
            return step
        return ((step // interval) + 1) * interval

    next_log = next_multiple(global_samples, args.log_every)
    next_eval = next_multiple(global_samples, args.eval_every_steps)
    next_save = next_multiple(global_samples, args.save_every)
    last_saved_step = 0

    epoch = 0
    data_iter = iter(train_loader)
    progress_bar = None
    if args.progress and accelerator.is_local_main_process:
        progress_bar = tqdm(total=args.max_steps, desc="Training", leave=True)
        if global_samples > 0:
            progress_bar.update(global_samples)

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

        kl_zero_steps = int(getattr(args, "kl_zero_steps", 0) or 0)
        kl_warmup_steps = int(getattr(args, "kl_warmup_steps", 0) or 0)
        if global_samples < kl_zero_steps:
            kl_weight = 0.0
        elif kl_warmup_steps <= kl_zero_steps or global_samples >= kl_warmup_steps:
            kl_weight = float(args.kl_weight)
        else:
            # Linear growth from kl_zero_steps to kl_warmup_steps
            progress = float(global_samples - kl_zero_steps) / float(
                kl_warmup_steps - kl_zero_steps
            )
            kl_weight = float(args.kl_weight) * progress

        with accelerator.accumulate(model):
            batch = batch.to(accelerator.device, non_blocking=True)
            # Normalize (no padding needed)
            batch = (batch - mean) / std

            with accelerator.autocast():
                recon, mu, logvar = model(batch)
                recon_loss = mse_loss(recon, batch)
                kl = kl_loss(mu, logvar)
                loss = recon_loss + kl_weight * kl

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
        kl_sum = accelerator.reduce(kl.detach() * batch_size_tensor, reduction="sum")

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
                        "checkpoint": f"vae_attention_checkpoint_step_{global_samples}.pt",
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
                    "wandb_run_id": wandb_run.id if wandb_run is not None else None,
                }
                checkpoint_path = (
                    output_dir / f"vae_attention_checkpoint_step_{global_samples}.pt"
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
                "wandb_run_id": wandb_run.id if wandb_run is not None else None,
            }
            checkpoint_path = (
                output_dir / f"vae_attention_checkpoint_step_{global_samples}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
