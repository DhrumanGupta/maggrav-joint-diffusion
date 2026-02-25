"""
Encode all samples from a Noddyverse Zarr store into VAE latents and
write them to a new Zarr store. Designed to run with `accelerate`.

Usage:
    accelerate launch src/encode_latents.py \
        --input_zarr /path/to/input.zarr \
        --output_zarr /path/to/output_latents.zarr \
        --checkpoint outputs_vae/vae_checkpoint_step_1000000.pt

Filtering:
    Pass --use_filter_cache to restrict encoding to informative samples selected
    by the pre-computed rock-count filter cache (same cache used during training).

    accelerate launch src/encode_latents.py \
        --input_zarr /path/to/input.zarr \
        --output_zarr /path/to/output_latents.zarr \
        --checkpoint outputs_vae/vae_checkpoint_step_1000000.pt \
        --use_filter_cache \
        --filter_cache_path /path/to/train_filter_stats.pt \
        --susc_active_threshold_log10 -1.5 \
        --min_active_frac 0.001 \
        --low_info_keep_prob 0.02

    The output Zarr will contain only the selected samples stored compactly
    (shape[0] == filtered_count, reindexed 0..filtered_count-1).
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import zarr
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from ..data.filtering import load_filter_stats_cache, select_indices_from_filter_stats
from ..models.vae import VAE3D, VAE3DConfig
from ..models.vae_attention import VAE3DAttention, VAE3DAttentionConfig
from ..utils.checkpoint import clean_state_dict, load_checkpoint
from ..utils.datasets import JointDensitySuscDataset
from ..utils.stats import load_stats

# Padding constants: original data is 200³, pad to 256³ for nice latent dimensions
ORIGINAL_SIZE = 200
PADDED_SIZE = 256
PAD_AMOUNT = (PADDED_SIZE - ORIGINAL_SIZE) // 2  # 28 on each side

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


def _preprocess_input(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    use_padding: bool,
) -> torch.Tensor:
    """Normalize input and optionally pad to 256³ for the convolutional VAE."""
    x = (x - mean) / std
    if use_padding:
        x = pad_to_256(x)
    return x


class _CompactSubset(Dataset):
    """Wrap a Subset so that __getitem__(i) returns (i, tensor) instead of
    (raw_zarr_index, tensor).  This makes filtered subsets behave identically
    to range-based subsets: the returned index IS the compact output position,
    so encode_and_save can write directly without any offset arithmetic."""

    def __init__(self, subset: Subset) -> None:
        self._subset = subset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, i: int) -> Tuple[int, torch.Tensor]:
        _, tensor = self._subset[i]
        return i, tensor


def _collate_with_indices(
    batch: list[Tuple[int, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices, tensors = zip(*batch)
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.stack(tensors, dim=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode VAE latents for all samples and write to Zarr."
    )
    parser.add_argument("--input_zarr", type=str, required=True)
    parser.add_argument("--output_zarr", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--save_logvar",
        action="store_true",
        help="Also save per-sample log-variance alongside latent means.",
    )
    parser.add_argument(
        "--copy_metadata",
        action="store_true",
        help="Copy samples_metadata array/attr from the input store to the output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output Zarr store if it already exists.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only process the first num_samples. If None, process all samples.",
    )

    # Filtering options (mirror latent_pca / train_vae_attention)
    parser.add_argument(
        "--use_filter_cache",
        action="store_true",
        help=(
            "Pre-filter raw samples using a pre-computed rock-count filter cache "
            "before encoding. Requires --filter_cache_path."
        ),
    )
    parser.add_argument(
        "--filter_cache_path",
        type=str,
        default=None,
        help="Path to the filter cache .pt file produced by train_vae_attention.",
    )
    parser.add_argument(
        "--susc_active_threshold_log10",
        type=float,
        default=-2.5,
        help="log10 susceptibility threshold above which a rock ID is considered active.",
    )
    parser.add_argument(
        "--min_active_frac",
        type=float,
        default=0.005,
        help="Minimum fraction of active voxels for a sample to be 'informative'.",
    )
    parser.add_argument(
        "--low_info_keep_prob",
        type=float,
        default=0.2,
        help="Probability to keep a low-information sample after filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic low-info sample selection.",
    )

    args = parser.parse_args()

    if args.use_filter_cache:
        if not args.filter_cache_path:
            parser.error(
                "--filter_cache_path is required when --use_filter_cache is set."
            )
        if not (0.0 <= args.min_active_frac <= 1.0):
            parser.error("--min_active_frac must be in [0, 1].")
        if not (0.0 <= args.low_info_keep_prob <= 1.0):
            parser.error("--low_info_keep_prob must be in [0, 1].")

    return args


def _infer_latent_shape(
    dataset: Dataset,
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    accelerator: Accelerator,
    use_padding: bool,
) -> Tuple[int, int, int, int, int]:
    """Run a single forward pass to determine latent tensor shape."""
    sample = dataset[0][1].unsqueeze(0).to(accelerator.device, non_blocking=True)
    sample = _preprocess_input(sample, mean, std, use_padding)
    with torch.no_grad():
        with accelerator.autocast():
            mu, _ = model.encode(sample)
    return (
        len(dataset),
        mu.shape[1],
        mu.shape[2],
        mu.shape[3],
        mu.shape[4],
    )


def _get_default_compressors() -> Optional[Tuple[object, ...]]:
    try:
        from zarr.codecs import Shuffle, Zstd

        # Use a Shuffle step followed by Zstd (level 5) for good lossless compression.
        # Zstd offers a good balance between compression ratio and speed.
        return (
            Shuffle(),
            Zstd(level=2),
        )
    except Exception as e:
        print(f"Warning: could not import Zstd/Shuffle ({e}), proceeding uncompressed.")
        return None


def _initialize_output_store(
    output_path: str,
    latent_shape: Tuple[int, int, int, int, int],
    save_logvar: bool,
    overwrite: bool,
    copy_metadata: bool,
    input_zarr: str,
    checkpoint: str,
    stats_path: Path,
    model_type: str,
    model_config_class: str,
    filter_info: Optional[Dict] = None,
) -> None:
    mode = "w" if overwrite else "a"
    compressors = _get_default_compressors()
    group = zarr.open_group(output_path, mode=mode)

    chunks = (1, latent_shape[1], latent_shape[2], latent_shape[3], latent_shape[4])

    def _ensure_array(
        name: str,
        shape: Tuple[int, ...],
        dtype: str,
        array_chunks: Tuple[int, ...],
        array_compressors: Optional[Tuple[object, ...]],
    ) -> None:
        if name in group:
            existing = group[name]
            if existing.shape != shape:
                raise ValueError(
                    f"Existing array {name!r} has shape {existing.shape}, "
                    f"expected {shape}."
                )
            return
        group.create_array(
            name,
            shape=shape,
            chunks=array_chunks,
            dtype=dtype,
            compressors=array_compressors,
        )

    _ensure_array(
        "latent_mu",
        shape=latent_shape,
        dtype="f4",
        array_chunks=chunks,
        array_compressors=compressors,
    )
    if save_logvar:
        _ensure_array(
            "latent_logvar",
            shape=latent_shape,
            dtype="f4",
            array_chunks=chunks,
            array_compressors=compressors,
        )

    # Optional metadata propagation
    if copy_metadata:
        src_group = zarr.open_group(input_zarr, mode="r")
        if "samples_metadata" in src_group:
            src_meta = src_group["samples_metadata"]
            src_compressors = (
                tuple(src_meta.compressors)
                if getattr(src_meta, "compressors", None)
                else None
            )
            if "samples_metadata" not in group:
                if filter_info is not None:
                    # Write only metadata rows for the selected indices (compact order).
                    src_data = src_meta[:]
                    filtered_data = src_data[filter_info["selected_indices"]]
                    group.create_array(
                        "samples_metadata",
                        data=filtered_data,
                        chunks=src_meta.chunks,
                        compressors=src_compressors,
                    )
                else:
                    group.create_array(
                        "samples_metadata",
                        data=src_meta[:],
                        chunks=src_meta.chunks,
                        compressors=src_compressors,
                    )
        elif "samples_metadata" in src_group.attrs:
            raw = src_group.attrs["samples_metadata"]
            if filter_info is not None:
                import json as _json

                parsed = _json.loads(raw) if isinstance(raw, str) else raw
                group.attrs["samples_metadata"] = [
                    parsed[i] for i in filter_info["selected_indices"]
                ]
            else:
                group.attrs["samples_metadata"] = raw

        # Propagate other attributes shallowly
        for key, value in src_group.attrs.items():
            if key == "samples_metadata":
                continue
            group.attrs[key] = value

    group.attrs["source_zarr"] = input_zarr
    group.attrs["checkpoint"] = checkpoint
    group.attrs["stats_path"] = str(stats_path)
    group.attrs["latent_shape"] = latent_shape
    group.attrs["model_type"] = model_type
    group.attrs["model_config_class"] = model_config_class

    # Record filter provenance when filtering was applied
    if filter_info is not None:
        group.attrs["filter_enabled"] = True
        group.attrs["filter_cache_path"] = filter_info["filter_cache_path"]
        group.attrs["filter_susc_active_threshold_log10"] = filter_info[
            "susc_active_threshold_log10"
        ]
        group.attrs["filter_min_active_frac"] = filter_info["min_active_frac"]
        group.attrs["filter_low_info_keep_prob"] = filter_info["low_info_keep_prob"]
        group.attrs["filter_seed"] = filter_info["seed"]
        group.attrs["filter_cache_sample_count"] = filter_info["cache_sample_count"]
        group.attrs["filter_selected_count"] = filter_info["selected_count"]
        group.attrs["filter_informative_count"] = filter_info["informative_count"]
        group.attrs["filter_low_info_count"] = filter_info["low_info_count"]
        group.attrs["filter_retained_low_info_count"] = filter_info[
            "retained_low_info_count"
        ]
        group.attrs["filter_source_indices"] = filter_info["selected_indices"]
    else:
        group.attrs["filter_enabled"] = False


@torch.no_grad()
def encode_and_save(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    latent_mu_store: zarr.Array,
    latent_logvar_store: Optional[zarr.Array],
    accelerator: Accelerator,
    use_padding: bool,
) -> None:
    model.eval()
    device = accelerator.device

    progress = tqdm(
        dataloader,
        desc="Encoding",
        disable=not accelerator.is_local_main_process,
    )

    for indices, batch in progress:
        batch = batch.to(device, non_blocking=True)
        batch = _preprocess_input(batch, mean, std, use_padding)

        with accelerator.autocast():
            _, mu, logvar = model(batch)

        mu_np = mu.detach().cpu().float().numpy()
        logvar_np = (
            logvar.detach().cpu().float().numpy() if logvar is not None else None
        )
        indices_np = indices.cpu().numpy()
        index_list = indices_np.tolist()

        latent_mu_store[index_list] = mu_np
        if latent_logvar_store is not None and logvar_np is not None:
            latent_logvar_store[index_list] = logvar_np


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision="bf16",
    )

    device = accelerator.device
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(
        checkpoint_path, device, [VAE3DConfig, VAE3DAttentionConfig]
    )

    stats_path = (
        Path(args.stats_path)
        if args.stats_path is not None
        else Path(checkpoint.get("stats_path", ""))
    )
    if not stats_path.exists():
        raise FileNotFoundError(
            "Stats file not found. Provide --stats_path or ensure checkpoint includes "
            f"stats_path. Got: {stats_path}"
        )

    stats = load_stats(stats_path)
    mean = stats["mean"].float().view(1, 2, 1, 1, 1).to(device)
    std = stats["std"].float().view(1, 2, 1, 1, 1).to(device)

    model_cfg = checkpoint["config"]
    if isinstance(model_cfg, VAE3DConfig):
        model: Union[VAE3D, VAE3DAttention] = VAE3D(model_cfg)
        use_padding = True
        model_type = "VAE3D"
    elif isinstance(model_cfg, VAE3DAttentionConfig):
        model = VAE3DAttention(model_cfg)
        use_padding = False
        model_type = "VAE3DAttention"
    else:
        raise ValueError(
            f"Unsupported VAE config type in checkpoint: {type(model_cfg).__name__}"
        )
    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)

    full_dataset = JointDensitySuscDataset(args.input_zarr, return_index=True)

    # ------------------------------------------------------------------
    # Optional pre-encode filtering
    # ------------------------------------------------------------------
    filter_info: Optional[Dict] = None

    if args.use_filter_cache:
        filter_cache_path = Path(args.filter_cache_path)
        filter_stats = load_filter_stats_cache(
            cache_path=filter_cache_path,
            expected_train_size=None,
            expected_format=3,
            strict=True,
        )
        selection = select_indices_from_filter_stats(
            filter_stats=filter_stats,
            susc_active_threshold_log10=float(args.susc_active_threshold_log10),
            min_active_frac=float(args.min_active_frac),
            low_info_keep_prob=float(args.low_info_keep_prob),
            seed=int(args.seed),
        )
        selected_indices: List[int] = selection["selected_indices"]
        if not selected_indices:
            raise ValueError(
                "Filter cache produced 0 selected samples. Check filter parameters."
            )

        # Apply --num_samples cap on top of the filtered list if requested
        if args.num_samples is not None:
            selected_indices = selected_indices[: args.num_samples]

        filter_info = {
            "selected_indices": selected_indices,
            "cache_sample_count": int(selection["cache_sample_count"]),
            "selected_count": len(selected_indices),
            "informative_count": int(selection["informative_count"]),
            "low_info_count": int(selection["low_info_count"]),
            "retained_low_info_count": int(selection["retained_low_info_count"]),
            "filter_cache_path": str(filter_cache_path),
            "susc_active_threshold_log10": float(args.susc_active_threshold_log10),
            "min_active_frac": float(args.min_active_frac),
            "low_info_keep_prob": float(args.low_info_keep_prob),
            "seed": int(args.seed),
        }

        # _CompactSubset re-exposes the subset with 0-based indices so that the
        # existing write logic works identically to the --num_samples path.
        dataset: Dataset = _CompactSubset(Subset(full_dataset, selected_indices))

        if accelerator.is_main_process:
            retention = len(selected_indices) / float(selection["cache_sample_count"])
            logger.info("Filter cache loaded: %s", filter_cache_path)
            logger.info(
                "Filter params | susc_threshold=%.3f | min_active_frac=%.5f | "
                "low_info_keep_prob=%.3f | seed=%d",
                args.susc_active_threshold_log10,
                args.min_active_frac,
                args.low_info_keep_prob,
                args.seed,
            )
            logger.info(
                "Filter selection | cache_samples=%d | informative=%d | "
                "low_info=%d | retained_low_info=%d | selected=%d (%.1f%%)",
                selection["cache_sample_count"],
                selection["informative_count"],
                selection["low_info_count"],
                selection["retained_low_info_count"],
                len(selected_indices),
                retention * 100.0,
            )
    elif args.num_samples is not None:
        num_samples = min(args.num_samples, len(full_dataset))
        dataset = Subset(full_dataset, range(num_samples))
    else:
        dataset = full_dataset

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_with_indices,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=4, persistent_workers=True)

    dataloader = DataLoader(**loader_kwargs)

    latent_shape = _infer_latent_shape(
        dataset, model, mean, std, accelerator, use_padding
    )
    if accelerator.is_main_process:
        logger.info("Latent shape will be %s", latent_shape)
        logger.info(
            "DDP: processes=%d | global samples=%d | batch_size=%d | steps=%d",
            accelerator.num_processes,
            len(dataset),
            args.batch_size,
            len(dataloader),
        )
        _initialize_output_store(
            args.output_zarr,
            latent_shape,
            save_logvar=args.save_logvar,
            overwrite=args.overwrite,
            copy_metadata=args.copy_metadata,
            input_zarr=args.input_zarr,
            checkpoint=args.checkpoint,
            stats_path=stats_path,
            model_type=model_type,
            model_config_class=type(model_cfg).__name__,
            filter_info=filter_info,
        )

    accelerator.wait_for_everyone()

    # Re-open output stores in append mode on each process
    out_group = zarr.open_group(args.output_zarr, mode="r+")
    latent_mu_store = out_group["latent_mu"]
    latent_logvar_store = (
        out_group["latent_logvar"] if "latent_logvar" in out_group else None
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    encode_and_save(
        model,
        dataloader,
        mean,
        std,
        latent_mu_store,
        latent_logvar_store,
        accelerator,
        use_padding,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done. Latents written to %s", args.output_zarr)


if __name__ == "__main__":
    main()
