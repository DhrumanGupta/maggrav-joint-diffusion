#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import zarr

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None
try:
    from zarr.codecs import Blosc, VLenUTF8Codec
except Exception:  # pragma: no cover - optional for older zarr versions
    Blosc = None
    VLenUTF8Codec = None


def _looks_like_zarr_store(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, ".zgroup")):
        return True
    if os.path.exists(os.path.join(path, ".zarray")):
        return True
    if os.path.exists(os.path.join(path, "zarr.json")):
        return True
    return False


T = TypeVar("T")


def _iter_with_progress(items: Iterable[T], desc: str) -> Iterable[T]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="store", file=sys.stderr)


def _collect_zarr_paths(input_path: str) -> List[str]:
    if _looks_like_zarr_store(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise SystemExit(f"Input path is not a zarr store or directory: {input_path}")
    candidates = []
    for entry in sorted(os.listdir(input_path)):
        entry_path = os.path.join(input_path, entry)
        if _looks_like_zarr_store(entry_path):
            candidates.append(entry_path)
    return candidates


def _validate_store(
    store: zarr.Group,
    path: str,
    expected: Optional[Dict[str, Tuple[Tuple[int, ...], np.dtype]]],
) -> Tuple[int, Dict[str, Tuple[Tuple[int, ...], np.dtype]]]:
    required = ("rock_types", "mag", "grv")
    for name in required:
        if name not in store:
            raise ValueError(f"Missing array {name!r} in {path}")
    rock = store["rock_types"]
    mag = store["mag"]
    grv = store["grv"]

    if rock.ndim != 4 or mag.ndim != 3 or grv.ndim != 3:
        raise ValueError(
            f"Unexpected dimensions in {path}: "
            f"rock_types={rock.shape} mag={mag.shape} grv={grv.shape}"
        )

    sample_count = int(rock.shape[0])
    if mag.shape[0] != sample_count or grv.shape[0] != sample_count:
        raise ValueError(
            f"Sample axis mismatch in {path}: "
            f"rock_types={rock.shape[0]} mag={mag.shape[0]} grv={grv.shape[0]}"
        )

    signature = {
        "rock_types": (rock.shape[1:], rock.dtype),
        "mag": (mag.shape[1:], mag.dtype),
        "grv": (grv.shape[1:], grv.dtype),
    }
    if expected is None:
        return sample_count, signature
    for name, (shape, dtype) in signature.items():
        exp_shape, exp_dtype = expected[name]
        if shape != exp_shape or dtype != exp_dtype:
            raise ValueError(
                f"Array mismatch for {name!r} in {path}: "
                f"shape={shape} dtype={dtype} expected_shape={exp_shape} "
                f"expected_dtype={exp_dtype}"
            )
    return sample_count, expected


def _copy_array(
    src: zarr.Array, dst: zarr.Array, dst_offset: int, batch_size: int
) -> None:
    total = int(src.shape[0])
    if total == 0:
        return
    step = batch_size if batch_size > 0 else 0
    if step <= 0:
        chunk0 = src.chunks[0] if src.chunks else 1
        step = int(chunk0) if isinstance(chunk0, int) and chunk0 > 0 else 1
    for start in range(0, total, step):
        end = min(total, start + step)
        dst[dst_offset + start : dst_offset + end] = src[start:end]


def _parse_metadata(raw: Optional[str], path: str, expected_count: int) -> List[str]:
    if not raw:
        raise ValueError(f"Missing samples_metadata in {path}")
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"samples_metadata is not a list in {path}")
    if len(parsed) != expected_count:
        raise ValueError(
            f"samples_metadata length mismatch in {path}: "
            f"{len(parsed)} != {expected_count}"
        )
    return [json.dumps(entry) for entry in parsed]


def _configure_blosc_threads(num_threads: int) -> None:
    try:
        import numcodecs.blosc as blosc
    except Exception:
        return
    if num_threads and num_threads > 0:
        blosc.set_nthreads(num_threads)


def _copy_store_worker(args: Tuple[str, str, int, int, int]) -> None:
    input_path, output_path, offset, batch_size, blosc_threads = args
    _configure_blosc_threads(blosc_threads)
    store = zarr.open_group(input_path, mode="r")
    out = zarr.open_group(output_path, mode="r+")
    _copy_array(store["rock_types"], out["rock_types"], offset, batch_size)
    _copy_array(store["mag"], out["mag"], offset, batch_size)
    _copy_array(store["grv"], out["grv"], offset, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate per-tar Zarr stores into one ML-friendly Zarr."
    )
    parser.add_argument("input", help="Directory containing Zarr stores or a Zarr.")
    parser.add_argument("output", help="Output Zarr store path.")
    parser.add_argument(
        "--chunk-3d",
        type=int,
        nargs=4,
        default=None,
        metavar=("S", "Z", "X", "Y"),
        help="Chunk shape for rock_types (sample, z, x, y).",
    )
    parser.add_argument(
        "--chunk-2d",
        type=int,
        nargs=3,
        default=None,
        metavar=("S", "X", "Y"),
        help="Chunk shape for mag/grv (sample, x, y).",
    )
    parser.add_argument(
        "--compressor",
        choices=("lz4", "zstd"),
        default="lz4",
        help="Blosc compressor to use for the concatenated Zarr.",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        default=1,
        help="Compression level for Blosc (lz4 ignores levels > 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Sample batch size when copying (0 = use input chunk size).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use for copying.",
    )
    parser.add_argument(
        "--blosc-threads",
        type=int,
        default=1,
        help="Blosc threads per worker process (avoid oversubscription).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    args = parser.parse_args()

    zarr_paths = _collect_zarr_paths(args.input)
    if not zarr_paths:
        raise SystemExit(f"No Zarr stores found under {args.input}")

    if os.path.exists(args.output):
        if not args.overwrite:
            raise SystemExit(f"Output path already exists: {args.output}")
        if os.path.isdir(args.output):
            shutil.rmtree(args.output)
        else:
            os.remove(args.output)

    expected: Optional[Dict[str, Tuple[Tuple[int, ...], np.dtype]]] = None
    zarr_infos: List[Tuple[str, int]] = []
    total_samples = 0
    for path in _iter_with_progress(zarr_paths, "Scanning zarrs"):
        store = zarr.open_group(path, mode="r")
        sample_count, expected = _validate_store(store, path, expected)
        zarr_infos.append((path, sample_count))
        total_samples += sample_count

    if expected is None or total_samples == 0:
        raise SystemExit("No samples found to concatenate.")

    rock_shape, rock_dtype = expected["rock_types"]
    mag_shape, mag_dtype = expected["mag"]
    grv_shape, grv_dtype = expected["grv"]
    chunk_3d = tuple(args.chunk_3d) if args.chunk_3d is not None else (1, *rock_shape)
    chunk_2d = tuple(args.chunk_2d) if args.chunk_2d is not None else (1, *mag_shape)
    if args.workers > 1 and (chunk_3d[0] != 1 or chunk_2d[0] != 1):
        raise SystemExit(
            "Parallel copy requires sample-axis chunk size of 1 to avoid "
            "chunk write conflicts. Use --workers 1 or set chunk sizes to 1."
        )

    if Blosc is None:
        raise SystemExit("Blosc codec unavailable; cannot create compressed Zarr.")
    if VLenUTF8Codec is None:
        raise SystemExit("VLenUTF8Codec unavailable; cannot store metadata strings.")
    shuffle = getattr(Blosc, "BITSHUFFLE", 2)
    compressor = Blosc(cname=args.compressor, clevel=args.clevel, shuffle=shuffle)

    out = zarr.open_group(args.output, mode="w")
    out.attrs["source_zarrs"] = zarr_paths
    out.attrs["total_samples"] = total_samples
    out.create_array(
        "rock_types",
        shape=(total_samples, *rock_shape),
        chunks=chunk_3d,
        dtype=rock_dtype,
        compressors=[compressor],
    )
    out.create_array(
        "mag",
        shape=(total_samples, *mag_shape),
        chunks=chunk_2d,
        dtype=mag_dtype,
        compressors=[compressor],
    )
    out.create_array(
        "grv",
        shape=(total_samples, *grv_shape),
        chunks=chunk_2d,
        dtype=grv_dtype,
        compressors=[compressor],
    )
    meta_array = out.create_array(
        "samples_metadata",
        shape=(total_samples,),
        dtype=str,
        serializer=VLenUTF8Codec(),
    )

    tasks = []
    offset = 0
    for path, sample_count in zarr_infos:
        tasks.append((path, args.output, offset, args.batch_size, args.blosc_threads))
        offset += sample_count

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for _ in _iter_with_progress(
                executor.map(_copy_store_worker, tasks), "Copying zarrs"
            ):
                pass
    else:
        for task in _iter_with_progress(tasks, "Copying zarrs"):
            _copy_store_worker(task)

    offset = 0
    for path, sample_count in _iter_with_progress(zarr_infos, "Writing metadata"):
        store = zarr.open_group(path, mode="r")
        raw_meta = store.attrs.get("samples_metadata")
        meta_strings = _parse_metadata(raw_meta, path, sample_count)
        meta_array[offset : offset + sample_count] = meta_strings
        offset += sample_count

    print(
        f"Concatenated {len(zarr_infos)} Zarr stores into {args.output} "
        f"({total_samples} samples)."
    )


if __name__ == "__main__":
    main()
