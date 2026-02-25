"""
Basic diagnostics for latent Zarr stores produced by encode_latents.

Reports how many samples are entirely zero versus nonzero in a latent array.
A sample is considered zero if every value in that sample is exactly zero.

Usage:
    python -m src.data.latent_dataset_diagnostics --zarr_path /path/to/latents.zarr
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr
from tqdm.auto import tqdm

_WORKER_GROUP = None
_WORKER_ARRAY = None


def _init_worker(zarr_path: str, array_name: str) -> None:
    global _WORKER_GROUP, _WORKER_ARRAY
    _WORKER_GROUP = zarr.open_group(zarr_path, mode="r")
    _WORKER_ARRAY = _WORKER_GROUP[array_name]


def _count_zero_samples_in_range(start: int, stop: int, batch_size: int) -> int:
    if _WORKER_ARRAY is None:
        raise RuntimeError("Worker array is not initialized.")

    zero_samples = 0
    for pos in range(start, stop, batch_size):
        end = min(pos + batch_size, stop)
        batch = np.asarray(_WORKER_ARRAY[pos:end])
        flat = batch.reshape(batch.shape[0], -1)
        sample_has_any_nonzero = np.any(flat, axis=1)
        zero_samples += int(flat.shape[0] - np.count_nonzero(sample_has_any_nonzero))
    return zero_samples


def _compute_auto_batch_size(array: zarr.Array) -> int:
    chunk0 = 1
    if array.chunks is not None and len(array.chunks) > 0:
        chunk0 = max(1, int(array.chunks[0]))

    per_sample_values = 1
    if array.ndim > 1:
        per_sample_values = int(np.prod(array.shape[1:], dtype=np.int64))
    bytes_per_sample = max(1, per_sample_values * int(array.dtype.itemsize))

    target_bytes = 64 * 1024 * 1024
    raw_batch = max(chunk0, target_bytes // bytes_per_sample)
    aligned = (raw_batch // chunk0) * chunk0
    return max(chunk0, aligned)


def _partition_ranges(total_samples: int, workers: int) -> list[tuple[int, int]]:
    shard = int(math.ceil(total_samples / workers))
    ranges: list[tuple[int, int]] = []
    for i in range(workers):
        start = i * shard
        stop = min(total_samples, start + shard)
        if start >= stop:
            continue
        ranges.append((start, stop))
    return ranges


def _resolve_workers(requested_workers: int, total_samples: int) -> int:
    if total_samples <= 0:
        return 1
    if requested_workers > 0:
        return max(1, min(requested_workers, total_samples))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, total_samples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report zero/nonzero sample counts for a latent array in a Zarr store."
        )
    )
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument(
        "--array_name",
        type=str,
        default="latent_mu",
        help="Array to inspect (default: latent_mu).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help=(
            "Samples per read in each worker process (0 = auto-tuned from array shape)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (0 = auto, typically all CPU cores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 0:
        raise ValueError("--batch_size must be >= 0.")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0.")

    group = zarr.open_group(args.zarr_path, mode="r")
    if args.array_name not in group:
        raise KeyError(f"Array {args.array_name!r} not found in {args.zarr_path}.")

    array = group[args.array_name]
    if array.ndim < 1:
        raise ValueError(f"Array {args.array_name!r} must have at least 1 dimension.")

    total_samples = int(array.shape[0])
    batch_size = args.batch_size if args.batch_size > 0 else _compute_auto_batch_size(array)
    workers = _resolve_workers(args.workers, total_samples)

    zero_samples = 0
    if workers == 1:
        _init_worker(args.zarr_path, args.array_name)
        zero_samples = _count_zero_samples_in_range(0, total_samples, batch_size)
    else:
        ranges = _partition_ranges(total_samples, workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(args.zarr_path, args.array_name),
        ) as executor:
            futures = [
                executor.submit(_count_zero_samples_in_range, start, stop, batch_size)
                for start, stop in ranges
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Diagnosing"
            ):
                zero_samples += int(future.result())

    nonzero_samples = total_samples - zero_samples
    zero_pct = (100.0 * zero_samples / total_samples) if total_samples > 0 else 0.0
    nonzero_pct = (
        (100.0 * nonzero_samples / total_samples) if total_samples > 0 else 0.0
    )

    print(f"zarr_path: {args.zarr_path}")
    print(f"array_name: {args.array_name}")
    print(f"total_samples: {total_samples}")
    print(f"workers: {workers}")
    print(f"batch_size: {batch_size}")
    print(f"zero_samples: {zero_samples} ({zero_pct:.2f}%)")
    print(f"nonzero_samples: {nonzero_samples} ({nonzero_pct:.2f}%)")


if __name__ == "__main__":
    main()
