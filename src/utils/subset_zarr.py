#!/usr/bin/env python3
"""
Extract a subset of samples from a Zarr store.

Usage:
    python -m src.utils.subset_zarr --input /path/to/input.zarr --output /path/to/output.zarr --num_samples 5000
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zarr

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _iter_with_progress(iterable, desc: str, total: Optional[int] = None):
    """Wrap an iterable with a progress bar if tqdm is available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="batch", file=sys.stderr)


def _copy_array_subset(
    src: zarr.Array,
    dst: zarr.Array,
    num_samples: int,
    batch_size: int = 100,
) -> None:
    """Copy the first num_samples from src to dst in batches."""
    for start in _iter_with_progress(
        range(0, num_samples, batch_size),
        desc=f"Copying {src.name}",
        total=(num_samples + batch_size - 1) // batch_size,
    ):
        end = min(start + batch_size, num_samples)
        dst[start:end] = src[start:end]


def get_array_config(arr: zarr.Array) -> Dict[str, Any]:
    """Extract configuration from an existing zarr array to replicate it."""
    config = {
        "chunks": arr.chunks,
        "dtype": arr.dtype,
    }
    # Handle compressor/compressors depending on zarr version
    if hasattr(arr, "compressors") and arr.compressors:
        config["compressors"] = arr.compressors
    elif hasattr(arr, "compressor") and arr.compressor:
        config["compressor"] = arr.compressor

    # Handle filters if present
    if hasattr(arr, "filters") and arr.filters:
        config["filters"] = arr.filters

    return config


def subset_zarr(
    input_path: str,
    output_path: str,
    num_samples: int,
    batch_size: int = 100,
    overwrite: bool = False,
) -> None:
    """
    Extract the first num_samples from a Zarr store to a new store.

    Args:
        input_path: Path to the input Zarr store.
        output_path: Path to the output Zarr store.
        num_samples: Number of samples to extract.
        batch_size: Batch size for copying arrays.
        overwrite: If True, overwrite the output path if it exists.
    """
    # Validate input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input Zarr store not found: {input_path}")

    # Handle output path
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Use --overwrite to replace."
            )
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)

    # Open input store
    src_group = zarr.open_group(input_path, mode="r")

    # Determine the total number of samples in the input
    # Try to find a reference array to get the sample count
    sample_count = None
    for name in src_group.array_keys():
        arr = src_group[name]
        if arr.ndim >= 1:
            sample_count = arr.shape[0]
            break

    if sample_count is None:
        raise ValueError("Could not determine sample count from input Zarr store.")

    if num_samples > sample_count:
        print(
            f"Warning: Requested {num_samples} samples but only {sample_count} available. "
            f"Using all {sample_count} samples.",
            file=sys.stderr,
        )
        num_samples = sample_count

    print(f"Extracting {num_samples} samples from {input_path} to {output_path}")

    # Create output store
    dst_group = zarr.open_group(output_path, mode="w")

    # Copy group-level attributes
    for key, value in src_group.attrs.items():
        # Handle samples_metadata attribute specially if it's a list
        if key == "samples_metadata" and isinstance(value, str):
            try:
                meta_list = json.loads(value)
                if isinstance(meta_list, list):
                    dst_group.attrs[key] = json.dumps(meta_list[:num_samples])
                    continue
            except json.JSONDecodeError:
                pass
        dst_group.attrs[key] = value

    # Update total_samples attribute if present
    if "total_samples" in dst_group.attrs:
        dst_group.attrs["total_samples"] = num_samples

    # Copy arrays
    for name in src_group.array_keys():
        src_arr = src_group[name]
        arr_config = get_array_config(src_arr)

        # Determine output shape (reduce first dimension to num_samples)
        if src_arr.ndim >= 1:
            out_shape = (num_samples,) + src_arr.shape[1:]
        else:
            out_shape = src_arr.shape

        # Adjust chunks if needed
        chunks = arr_config.get("chunks")
        if chunks and len(chunks) == len(out_shape):
            # Make sure chunk size doesn't exceed array size
            chunks = tuple(min(c, s) for c, s in zip(chunks, out_shape))
            arr_config["chunks"] = chunks

        # Create destination array
        dst_arr = dst_group.create_array(
            name,
            shape=out_shape,
            **{k: v for k, v in arr_config.items() if k not in ["shape"]},
        )

        # Copy data
        if src_arr.ndim >= 1:
            _copy_array_subset(src_arr, dst_arr, num_samples, batch_size)
        else:
            dst_arr[...] = src_arr[...]

    # Handle samples_metadata array if present
    if "samples_metadata" in src_group and isinstance(
        src_group["samples_metadata"], zarr.Array
    ):
        # Already handled in the array loop above
        pass

    print(f"Successfully extracted {num_samples} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a subset of samples from a Zarr store."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input Zarr store.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the output Zarr store.",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        required=True,
        help="Number of samples to extract from the beginning.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=100,
        help="Batch size for copying arrays (default: 100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    args = parser.parse_args()

    subset_zarr(
        input_path=args.input,
        output_path=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
