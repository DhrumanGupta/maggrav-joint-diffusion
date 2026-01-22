"""
Encode all samples from a Noddyverse Zarr store into VAE latents and
write them to a new Zarr store. Designed to run with `accelerate`.

Usage:
    accelerate launch src/encode_latents.py \
        --input_zarr /path/to/input.zarr \
        --output_zarr /path/to/output_latents.zarr \
        --checkpoint outputs_vae/vae_checkpoint_step_1000000.pt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import zarr
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from ..models.vae import VAE3D, VAE3DConfig
from ..utils.checkpoint import clean_state_dict, load_checkpoint
from ..utils.datasets import JointDensitySuscDataset
from ..utils.stats import load_stats

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    return parser.parse_args()


def _infer_latent_shape(
    dataset: Dataset,
    model: VAE3D,
    mean: torch.Tensor,
    std: torch.Tensor,
    accelerator: Accelerator,
) -> Tuple[int, int, int, int, int]:
    """Run a single forward pass to determine latent tensor shape."""
    sample = dataset[0][1].unsqueeze(0).to(accelerator.device, non_blocking=True)
    sample = (sample - mean) / std
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
        from zarr.codecs import Blosc as ZarrBlosc

        return (ZarrBlosc(cname="zstd", clevel=3, shuffle=ZarrBlosc.BITSHUFFLE),)
    except Exception:
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
                group.create_array(
                    "samples_metadata",
                    data=src_meta,
                    chunks=src_meta.chunks,
                    dtype=src_meta.dtype,
                    compressors=src_compressors,
                )
        elif "samples_metadata" in src_group.attrs:
            group.attrs["samples_metadata"] = src_group.attrs["samples_metadata"]

        # Propagate other attributes shallowly
        for key, value in src_group.attrs.items():
            if key == "samples_metadata":
                continue
            group.attrs[key] = value

    group.attrs["source_zarr"] = input_zarr
    group.attrs["checkpoint"] = checkpoint
    group.attrs["stats_path"] = str(stats_path)
    group.attrs["latent_shape"] = latent_shape


@torch.no_grad()
def encode_and_save(
    model: VAE3D,
    dataloader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    latent_mu_store: zarr.Array,
    latent_logvar_store: Optional[zarr.Array],
    accelerator: Accelerator,
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
        batch = (batch - mean) / std

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
            "Stats file not found. Provide --stats_path or ensure checkpoint includes "
            f"stats_path. Got: {stats_path}"
        )

    stats = load_stats(stats_path)
    mean = stats["mean"].float().view(1, 2, 1, 1, 1).to(device)
    std = stats["std"].float().view(1, 2, 1, 1, 1).to(device)

    model_cfg = checkpoint["config"]
    model = VAE3D(model_cfg)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = JointDensitySuscDataset(args.input_zarr, return_index=True)
    sampler: Optional[DistributedSampler] = None
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
        sampler.set_epoch(0)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_with_indices,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

    dataloader = DataLoader(**loader_kwargs)

    latent_shape = _infer_latent_shape(dataset, model, mean, std, accelerator)
    if accelerator.is_main_process:
        logger.info("Latent shape will be %s", latent_shape)
        per_rank_samples = sampler.num_samples if sampler is not None else len(dataset)
        logger.info(
            "DDP: processes=%d | per-rank samples=%d | batch_size=%d | steps=%d",
            accelerator.num_processes,
            per_rank_samples,
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
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done. Latents written to %s", args.output_zarr)


if __name__ == "__main__":
    main()
