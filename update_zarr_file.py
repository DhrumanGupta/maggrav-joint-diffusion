#!/usr/bin/env python3
"""
rezarr_v3_shard_n.py

Rewrite an existing (directory) Zarr into a new Zarr v3 layout that is much more
network-filesystem friendly by introducing *sharding* that is “fat only along N”
(axis 0), while keeping your existing chunking unchanged.

Your case:
  shape = (N, 2, 200, 200, 200)
  training reads along axis 0 (N)

This script:
  - Recursively copies groups/arrays + attrs
  - Keeps CHUNKS identical to the source (unless you override)
  - Sets SHARDS = (chunks[0] * shard_factor_n, chunks[1], chunks[2], ...)
    i.e. sharding only along N to match your access pattern
  - Copies data shard-by-shard (writes align with shard units)
  - Optionally consolidates metadata (helps open/scan speed on network FS)

Usage:
  python rezarr_v3_shard_n.py \
    --src /path/to/old.zarr \
    --dst /path/to/new_sharded.zarr \
    --shard-factor-n 64 \
    --consolidate \
    --overwrite

Notes:
  - This is designed for zarr-python v3, but includes fallbacks for minor API differences.
  - Downstream reads do NOT need to change if you keep paths/shape/dtype the same.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import zarr  # type: ignore
except Exception as e:
    print(
        "ERROR: Could not import zarr. Install zarr-python v3 in this environment.",
        file=sys.stderr,
    )
    raise


# -----------------------------
# Helpers
# -----------------------------


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} B"


def _iter_blocks(
    shape: Tuple[int, ...], block: Tuple[int, ...]
) -> Iterable[Tuple[slice, ...]]:
    """
    Yield N-D slices that tile the array by 'block' sizes.
    """
    nd = len(shape)
    grid = [range(_ceil_div(shape[d], block[d])) for d in range(nd)]

    def rec(d: int, cur: List[slice]):
        if d == nd:
            yield tuple(cur)
            return
        for i in grid[d]:
            start = i * block[d]
            stop = min(shape[d], start + block[d])
            cur.append(slice(start, stop))
            yield from rec(d + 1, cur)
            cur.pop()

    yield from rec(0, [])


def _safe_getattr(obj: Any, names: List[str]) -> Any:
    """
    Return the first successfully-retrieved attribute from 'names', else None.
    Handles properties that raise exceptions on access (e.g., Zarr v3 .compressor).
    """
    for n in names:
        try:
            return getattr(obj, n)
        except (AttributeError, TypeError, NotImplementedError):
            continue
    return None


def _choose_shards_n_only(
    shape: Tuple[int, ...], chunks: Tuple[int, ...], shard_factor_n: int
) -> Tuple[int, ...]:
    """
    Make shards larger only along axis 0 (N). Other axes keep shard == chunk.

    This matches your training access pattern (reading batches along N).
    """
    if len(shape) != len(chunks):
        raise ValueError(f"shape rank {len(shape)} != chunks rank {len(chunks)}")

    shard_factor_n = max(1, int(shard_factor_n))
    n_shard = min(shape[0], chunks[0] * shard_factor_n)

    # Ensure shard is a multiple of chunk for clean packing.
    n_shard = max(chunks[0], (n_shard // chunks[0]) * chunks[0])
    if n_shard <= 0:
        n_shard = chunks[0]

    return (n_shard,) + tuple(chunks[1:])


def _estimate_block_bytes(block_shape: Tuple[int, ...], dtype: np.dtype) -> int:
    n = 1
    for d in block_shape:
        n *= int(d)
    return int(n) * int(dtype.itemsize)


def _slice_shape(slc: Tuple[slice, ...]) -> Tuple[int, ...]:
    out = []
    for s in slc:
        start = 0 if s.start is None else int(s.start)
        stop = int(s.stop)
        out.append(stop - start)
    return tuple(out)


# -----------------------------
# Copy logic
# -----------------------------


@dataclass
class Plan:
    shard_factor_n: int
    overwrite: bool
    consolidate: bool
    dry_run: bool
    max_block_bytes: int  # safety to avoid reading enormous shards into RAM
    workers: int  # number of parallel workers for block copying


def _log(msg: str) -> None:
    print(msg, flush=True)


def _open_group(store: str, mode: str):
    """
    Open a group with best-effort compatibility across zarr v3 variants.
    """
    # Common case in zarr-python: open_group(store=path, mode=...)
    try:
        return zarr.open_group(store=store, mode=mode)
    except TypeError:
        # Some variants accept path positional
        return zarr.open_group(store, mode=mode)


def _list_group_children(src_g) -> List[Tuple[str, Any]]:
    """
    Return list of (name, obj) children for a group across zarr APIs.
    """
    if hasattr(src_g, "items"):
        try:
            return list(src_g.items())
        except Exception:
            pass

    if hasattr(src_g, "members"):
        try:
            members = src_g.members()
            if isinstance(members, dict):
                return list(members.items())
            out: List[Tuple[str, Any]] = []
            for m in members:
                if isinstance(m, tuple) and len(m) == 2:
                    out.append((m[0], m[1]))
                elif hasattr(m, "name"):
                    key = getattr(m, "basename", None) or str(m.name).split("/")[-1]
                    out.append((key, m))
            if out:
                return out
        except Exception:
            pass

    children: List[Tuple[str, Any]] = []
    if hasattr(src_g, "array_keys"):
        try:
            for k in src_g.array_keys():
                children.append((k, src_g[k]))
        except Exception:
            pass
    if hasattr(src_g, "group_keys"):
        try:
            for k in src_g.group_keys():
                children.append((k, src_g[k]))
        except Exception:
            pass
    if children:
        return children

    if hasattr(src_g, "keys"):
        try:
            return [(k, src_g[k]) for k in src_g.keys()]
        except Exception:
            pass

    raise AttributeError("Group object does not expose iterable children")


def _create_group(parent, name: str, overwrite: bool):
    """
    Create (or get) subgroup.
    """
    # zarr v3 group API typically has create_group(name, overwrite=...)
    try:
        return parent.create_group(name, overwrite=overwrite)
    except TypeError:
        # fallback: require exists_ok pattern
        if overwrite and name in parent:
            del parent[name]
        return parent.require_group(name)


def _create_array_like(
    dst_parent,
    name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunks: Tuple[int, ...],
    shards: Tuple[int, ...],
    fill_value: Any,
    src_array: Any,
    overwrite: bool,
):
    """
    Create destination array while attempting to preserve codec/compressor/filter settings
    across zarr versions.

    We try multiple keyword combinations and fall back to minimal args if needed.
    """
    # Collect optional settings from source array, but don't assume API names.
    # zarr v3 uses codecs; zarr v2 used compressor/filters.
    create_kwargs: Dict[str, Any] = dict(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        overwrite=overwrite,
    )

    # Sharding: zarr v3 supports `shards=...`
    create_kwargs["shards"] = shards

    # Fill value
    if fill_value is not None:
        create_kwargs["fill_value"] = fill_value

    # Try to preserve codecs / compressors / filters where possible
    codecs = _safe_getattr(src_array, ["codecs"])
    if codecs is not None:
        create_kwargs["codecs"] = codecs

    compressors = _safe_getattr(src_array, ["compressors"])
    filters = _safe_getattr(src_array, ["filters"])

    # Some builds accept compressors/filters
    if compressors is not None:
        create_kwargs["compressors"] = compressors
    if filters is not None:
        create_kwargs["filters"] = filters

    # Create with progressive fallback
    # 1) Full kwargs
    try:
        return dst_parent.create_array(**create_kwargs)
    except TypeError:
        pass

    # 2) Remove codecs-related fields (if unsupported)
    for k in ["codecs", "compressors", "filters"]:
        create_kwargs.pop(k, None)
    try:
        return dst_parent.create_array(**create_kwargs)
    except TypeError:
        pass

    # 3) Minimal create: no shards (if shards kw not supported in their install)
    #    This is not ideal for your goal, but better than crashing. We warn loudly.
    _log(
        f"WARNING: 'shards=' not accepted by this zarr installation for array '{name}'. "
        f"Falling back to unsharded array (file count may remain high)."
    )
    create_kwargs.pop("shards", None)
    return dst_parent.create_array(**create_kwargs)


def copy_group(
    src_g, dst_g, plan: Plan, src_store: str, dst_store: str, group_name: str = ""
) -> None:
    # Copy group attributes
    try:
        dst_g.attrs.update(dict(src_g.attrs))
    except Exception:
        # attrs API differences
        for k in src_g.attrs:
            dst_g.attrs[k] = src_g.attrs[k]

    # Get list of children for progress tracking
    children = _list_group_children(src_g)
    if not children:
        return

    # Create progress bar for group children
    group_desc = f"Processing {group_name}" if group_name else "Processing group"
    progress_bar = tqdm(
        children,
        desc=group_desc,
        unit="item",
        leave=False,
    )

    # Iterate children
    for key, obj in progress_bar:
        progress_bar.set_postfix(item=key)
        # zarr arrays generally have .shape and .dtype
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            copy_array(obj, dst_g, key, plan, src_store, dst_store)
        else:
            # assume group-like
            child_dst = _create_group(dst_g, key, overwrite=plan.overwrite)
            child_path = f"{group_name}/{key}" if group_name else key
            copy_group(
                obj, child_dst, plan, src_store, dst_store, group_name=child_path
            )


def _copy_blocks_batch_worker(
    src_store: str,
    dst_store: str,
    array_path: str,
    slc_indices_batch: List[Tuple[Tuple[int, int], ...]],
    max_block_bytes: int,
    worker_id: int,
    total_workers: int,
    array_name: str,
) -> int:
    """
    Worker function for copying a batch of blocks in a separate process.
    Opens zarr stores once and processes all assigned blocks.
    Each worker displays its own progress bar.
    Returns the number of blocks processed.
    """
    # Open arrays once for all blocks in this worker
    src_root = zarr.open_group(store=src_store, mode="r")
    dst_root = zarr.open_group(store=dst_store, mode="a")

    src_a = src_root[array_path]
    dst_a = dst_root[array_path]
    dtype = np.dtype(src_a.dtype)

    # Create progress bar for this worker at the correct position
    progress_bar = tqdm(
        total=len(slc_indices_batch),
        desc=f"Worker {worker_id}/{total_workers} ({array_name})",
        unit="block",
        position=worker_id,
        leave=False,
    )

    for slc_indices in slc_indices_batch:
        # Reconstruct slice objects from indices
        slc = tuple(slice(start, stop) for start, stop in slc_indices)

        block_shape = _slice_shape(slc)
        block_bytes = _estimate_block_bytes(block_shape, dtype)

        if block_bytes > max_block_bytes:
            progress_bar.close()
            raise MemoryError(
                f"Block read too large: {block_shape} => {_human_bytes(block_bytes)} "
                f"exceeds max_block_bytes={_human_bytes(max_block_bytes)}. "
                f"Lower --shard-factor-n."
            )

        data = src_a[slc]
        dst_a[slc] = data
        progress_bar.update(1)
        progress_bar.set_postfix(
            block_shape=block_shape,
            block_size=_human_bytes(block_bytes),
        )

    progress_bar.close()
    return len(slc_indices_batch)


def _partition_list(lst: List, n_parts: int) -> List[List]:
    """Split list into n_parts roughly equal chunks."""
    if n_parts <= 0:
        return [lst]
    k, m = divmod(len(lst), n_parts)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_parts)
    ]


def copy_array(
    src_a, dst_parent, name: str, plan: Plan, src_store: str, dst_store: str
) -> None:
    shape = tuple(int(x) for x in src_a.shape)
    dtype = np.dtype(src_a.dtype)

    # Chunks:
    # zarr arrays may have src_a.chunks (tuple) or a chunk_grid
    chunks = _safe_getattr(src_a, ["chunks"])
    if chunks is None:
        # last resort: treat as single chunk
        chunks = shape
    chunks = tuple(int(x) for x in chunks)

    # Choose shards: fat on axis 0 only
    shards = _choose_shards_n_only(shape, chunks, plan.shard_factor_n)

    # Estimate shard RAM footprint (worst-case read into memory in our loop)
    shard_bytes = _estimate_block_bytes(shards, dtype)
    if shard_bytes > plan.max_block_bytes:
        _log(
            f"WARNING: Shard for '{name}' would be {_human_bytes(shard_bytes)} in RAM, "
            f"which exceeds --max-block-bytes={_human_bytes(plan.max_block_bytes)}.\n"
            f"  Reducing shard_factor_n is recommended.\n"
            f"  Current: chunks={chunks}, shards={shards}"
        )

    # Create dest array
    fill_value = _safe_getattr(src_a, ["fill_value"])
    if plan.dry_run:
        _log(
            f"[dry-run] would create array '{name}': shape={shape} dtype={dtype} chunks={chunks} shards={shards}"
        )
        return

    dst_a = _create_array_like(
        dst_parent=dst_parent,
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        fill_value=fill_value,
        src_array=src_a,
        overwrite=plan.overwrite,
    )

    # Copy array attrs
    try:
        dst_a.attrs.update(dict(src_a.attrs))
    except Exception:
        for k in src_a.attrs:
            dst_a.attrs[k] = src_a.attrs[k]

    # Copy data shard-by-shard (block = shards)
    total_blocks = 1
    for s, b in zip(shape, shards):
        total_blocks *= _ceil_div(s, b)

    # Get array path for worker processes
    array_path = dst_a.path if hasattr(dst_a, "path") else name
    _log(
        f"[array] {array_path}: "
        f"shape={shape} chunks={chunks} shards={shards} blocks={total_blocks} "
        f"(~{_human_bytes(shard_bytes)} per full shard)"
    )

    # Prepare list of all block slices as serializable indices
    all_slices = list(_iter_blocks(shape, shards))
    # Convert slices to (start, stop) tuples for pickling
    all_slice_indices = [
        tuple((s.start or 0, s.stop) for s in slc) for slc in all_slices
    ]

    if plan.workers > 1:
        # Partition blocks among workers - each worker gets a batch
        batches = _partition_list(all_slice_indices, plan.workers)
        # Filter out empty batches
        batches = [b for b in batches if b]
        total_workers = len(batches)

        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            futures = [
                executor.submit(
                    _copy_blocks_batch_worker,
                    src_store,
                    dst_store,
                    array_path,
                    batch,
                    plan.max_block_bytes,
                    worker_id,
                    total_workers,
                    name,
                )
                for worker_id, batch in enumerate(batches)
            ]
            # Wait for all workers to complete
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Print newlines to move past the worker progress bars
        print("\n" * total_workers, end="", flush=True)
    else:
        # Sequential block copying with single progress bar
        progress_bar = tqdm(
            total=total_blocks,
            desc=f"Copying {name}",
            unit="block",
            leave=False,
        )
        for slc in all_slices:
            block_shape = _slice_shape(slc)
            block_bytes = _estimate_block_bytes(block_shape, dtype)

            if block_bytes > plan.max_block_bytes:
                progress_bar.close()
                raise MemoryError(
                    f"Block read too large for '{name}': {block_shape} => {_human_bytes(block_bytes)} "
                    f"exceeds --max-block-bytes={_human_bytes(plan.max_block_bytes)}. "
                    f"Lower --shard-factor-n."
                )

            data = src_a[slc]
            dst_a[slc] = data
            progress_bar.update(1)
            progress_bar.set_postfix(
                block_shape=block_shape,
                block_size=_human_bytes(block_bytes),
            )
        progress_bar.close()


def maybe_consolidate(dst_store: str) -> None:
    """
    Consolidate metadata if supported by the installed zarr version.
    This helps opening datasets on network filesystems.
    """
    # Different zarr versions expose different names/signatures.
    for fn_name in ["consolidate_metadata", "consolidate_metadata_async"]:
        fn = getattr(zarr, fn_name, None)
        if fn is None:
            continue
        try:
            fn(dst_store)
            _log("[meta] consolidated metadata written")
            return
        except TypeError:
            # Some variants want a store object rather than a path
            try:
                store_obj = zarr.storage.LocalStore(dst_store)  # type: ignore[attr-defined]
                fn(store_obj)
                _log("[meta] consolidated metadata written")
                return
            except Exception:
                pass
        except Exception as e:
            _log(f"[meta] consolidation attempted but failed: {e!r}")
            return

    _log("[meta] consolidation not available in this zarr installation; skipping")


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rewrite Zarr to Zarr v3 sharded layout optimized for axis-0 (N) scanning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src", required=True, help="Source Zarr store path (directory store)"
    )
    p.add_argument(
        "--dst", required=True, help="Destination Zarr store path (directory store)"
    )
    p.add_argument(
        "--shard-factor-n",
        type=int,
        default=64,
        help="Shard size along N = chunks[0] * shard_factor_n (other dims shard==chunk)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing destination nodes"
    )
    p.add_argument(
        "--consolidate",
        action="store_true",
        help="Write consolidated metadata at the end",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing data",
    )
    p.add_argument(
        "--max-block-bytes",
        type=str,
        default="2GiB",
        help="Safety limit for how much data (approx) can be loaded per shard/block into RAM. Examples: 512MiB, 2GiB",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for copying blocks. Use more workers to speed up on multi-core systems.",
    )
    return p.parse_args()


def parse_bytes(s: str) -> int:
    s = s.strip()
    units = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
    }
    # split numeric + suffix
    num = ""
    suf = ""
    for ch in s:
        if (ch.isdigit() or ch == ".") and suf == "":
            num += ch
        else:
            suf += ch
    num = num.strip()
    suf = suf.strip() or "B"
    if suf not in units:
        raise ValueError(
            f"Unrecognized byte suffix '{suf}'. Use one of: {', '.join(units.keys())}"
        )
    return int(float(num) * units[suf])


def main() -> None:
    args = parse_args()
    max_block_bytes = parse_bytes(args.max_block_bytes)

    # Default to number of CPUs if workers not specified or set to 0
    workers = args.workers
    if workers <= 0:
        import multiprocessing

        workers = multiprocessing.cpu_count()

    plan = Plan(
        shard_factor_n=max(1, int(args.shard_factor_n)),
        overwrite=bool(args.overwrite),
        consolidate=bool(args.consolidate),
        dry_run=bool(args.dry_run),
        max_block_bytes=max_block_bytes,
        workers=workers,
    )

    _log(f"[cfg] src={args.src}")
    _log(f"[cfg] dst={args.dst}")
    _log(f"[cfg] shard_factor_n={plan.shard_factor_n}  (shards only along axis 0)")
    _log(
        f"[cfg] overwrite={plan.overwrite} consolidate={plan.consolidate} dry_run={plan.dry_run}"
    )
    _log(f"[cfg] max_block_bytes={_human_bytes(plan.max_block_bytes)}")
    _log(f"[cfg] workers={plan.workers}")

    # Open source root group
    src_root = _open_group(args.src, mode="r")

    # Create/open destination root group
    if plan.dry_run:
        dst_root = None
    else:
        # mode="a" to create if missing, allow writing
        dst_root = _open_group(args.dst, mode="a")

        # If overwrite requested and destination already has content, we don't blindly delete
        # the whole store (dangerous). We rely on per-node overwrite semantics.
        # If you truly want a clean slate, delete the destination directory yourself first.

    _log("[root] copying hierarchy...")
    if plan.dry_run:
        # Use a dummy destination group reference: we just won't actually create arrays
        copy_group(src_root, src_root, plan, args.src, args.dst, group_name="root")  # type: ignore[arg-type]
    else:
        copy_group(src_root, dst_root, plan, args.src, args.dst, group_name="root")  # type: ignore[arg-type]

    if plan.consolidate and not plan.dry_run:
        maybe_consolidate(args.dst)

    _log("[done]")


if __name__ == "__main__":
    main()
