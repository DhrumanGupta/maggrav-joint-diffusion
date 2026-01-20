import argparse
import glob
import gzip
import io
import json
import math
import os
import re
import shutil
import tarfile
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from zarr.codecs import Blosc

# ----------------------------
# Parsing helpers
# ----------------------------


def _get_rss_mb() -> Optional[float]:
    """Return current RSS in MB if available (Linux), otherwise None."""
    try:
        with open("/proc/self/statm", "r") as f:
            rss_pages = int(f.read().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return rss_pages * page_size / (1024 * 1024)
    except Exception:
        return None


def _format_rss() -> str:
    rss = _get_rss_mb()
    if rss is None:
        return "rss=unknown"
    return f"rss={rss:.1f}MB"


def _open_gzip_text(source: Union[str, io.BufferedIOBase]):
    """Context manager to yield a text handle from a gzip file path or file-like."""
    if isinstance(source, str):
        return gzip.open(source, "rt")
    else:
        gz = gzip.GzipFile(fileobj=source, mode="rb")
        return io.TextIOWrapper(gz)


def load_gz_array(
    source: Union[str, io.BufferedIOBase],
    skiprows: int = 0,
    profile: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a gzipped text array, skipping optional header rows."""
    t0 = time.perf_counter() if profile else 0

    with _open_gzip_text(source) as f:
        for _ in range(skiprows):
            f.readline()
        text = f.read()

    if profile:
        t_read = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Tab-separated (g12) vs space/newline (mag/grv)
    if "\t" in text:
        text = text.replace("\t", " ")

    if profile:
        t_replace = time.perf_counter() - t0
        t0 = time.perf_counter()

    arr = np.fromstring(text, dtype=dtype, sep=" ")

    if profile:
        t_parse = time.perf_counter() - t0
        print(
            f"    load_gz: read={t_read:.3f}s replace={t_replace:.3f}s parse={t_parse:.3f}s (size={arr.size})"
        )

    return arr


def parse_g00_properties(
    g00_source: Union[str, io.BufferedIOBase],
    content: Optional[str] = None,
) -> Dict[int, Tuple[float, float]]:
    """Parse rock-code -> (density, susceptibility) from .g00.gz or text content."""
    props: Dict[int, Tuple[float, float]] = {}
    current_id: Optional[int] = None
    density: Optional[float] = None
    susc: Optional[float] = None

    if content is not None:
        lines = content.splitlines()
    else:
        with _open_gzip_text(g00_source) as f:
            lines = f.read().splitlines()

    for raw in lines:
        line = raw.strip()

        if line.startswith("ROCK DEFINITION"):
            match = re.search(r"=\s*(\d+)\s*$", line)
            if match:
                current_id = int(match.group(1))
                density = None
                susc = None

        elif "Density" in line and "=" in line and "CALCULATED" not in line:
            try:
                density = float(line.split("=")[1].strip())
            except (IndexError, ValueError):
                pass

        elif line.startswith("Sus") and "=" in line and "SUSCEPTIBILITY" not in line:
            try:
                susc = float(line.split("=")[1].strip())
            except (IndexError, ValueError):
                pass

        if current_id is not None and density is not None and susc is not None:
            props[current_id] = (density, susc)
            current_id, density, susc = None, None, None

    return props


def parse_g00_geometry(
    g00_source: Union[str, io.BufferedIOBase],
    content: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse geometry + field parameters from .g00.gz or text content (with sensible defaults)."""
    geom = {
        "num_layers": 200,
        "layer_dims": (200, 200),  # (nx, ny)
        "cube_size": 20.0,
        "mag_inclination": 90.0,
        "mag_intensity": 50000.0,
        "mag_declination": 0.0,
        "upper_corner": (-12000.0, -12000.0, 4000.0),
        "lower_corner": (-8000.0, -8000.0, 0.0),
    }

    try:
        if content is not None:
            lines = content.splitlines()
        else:
            with _open_gzip_text(g00_source) as f:
                lines = f.read().splitlines()

        for raw in lines:
            line = raw.strip()
            if line.startswith("NUMBER OF LAYERS"):
                geom["num_layers"] = int(line.split("=")[1].strip())
            elif "LAYER 1 DIMENSIONS" in line:
                dims = line.split("=")[1].strip().split()
                geom["layer_dims"] = (int(dims[0]), int(dims[1]))
            elif "CUBE SIZE FOR LAYER 1" in line:
                geom["cube_size"] = float(line.split("=")[1].strip())
            elif "INCLINATION OF EARTH MAG FIELD" in line:
                geom["mag_inclination"] = float(line.split("=")[1].strip())
            elif "INTENSITY OF EARTH MAG FIELD" in line:
                geom["mag_intensity"] = float(line.split("=")[1].strip())
            elif "DECLINATION" in line and "VOL" in line:
                try:
                    geom["mag_declination"] = float(line.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif "UPPER SW CORNER" in line:
                coords = line.split("=")[1].strip().split()
                geom["upper_corner"] = tuple(float(c) for c in coords)
            elif "LOWER NE CORNER" in line:
                coords = line.split("=")[1].strip().split()
                geom["lower_corner"] = tuple(float(c) for c in coords)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise RuntimeError("Failed to parse geometry from g00 source") from exc

    return geom


def parse_g00(
    g00_source: Union[str, io.BufferedIOBase],
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, Any]]:
    """Read g00 once and return (properties, geometry)."""
    with _open_gzip_text(g00_source) as f:
        content = f.read()
    props = parse_g00_properties(g00_source, content=content)
    geom = parse_g00_geometry(g00_source, content=content)
    return props, geom


# ----------------------------
# Loading helpers
# ----------------------------


def load_rock_types(
    g12_source: Union[str, io.BufferedIOBase],
    geom: Dict[str, Any],
    profile: bool = False,
) -> np.ndarray:
    """Load and reshape rock-type volume; orientation matches Noddyverse viewer."""
    nx, ny = geom["layer_dims"]
    nz = geom["num_layers"]
    # Parse directly as int32 - faster than float32 for integer data
    arr = load_gz_array(g12_source, skiprows=0, profile=profile, dtype=np.int32)
    expected = nx * ny * nz
    if arr.size != expected:
        raise ValueError(
            f"Unexpected g12 size {arr.size}, expected {expected} from geometry."
        )

    # Flatten order from C writer: for each z (top->bottom), for y rows, x cols.
    # Reshape to (z, y, x), then transpose to (z, x, y) as used in noddyverse.py.
    volume = arr.reshape((nz, ny, nx)).transpose(0, 2, 1)
    max_id = int(volume.max())
    if max_id > 255:
        source_label = g12_source if isinstance(g12_source, str) else "g12 stream"
        raise ValueError(
            f"rock_types max id {max_id} exceeds uint8 range (0-255) in {source_label}"
        )
    return volume


def map_properties(
    rock_types: np.ndarray, properties: Dict[int, Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Map rock codes to density & susceptibility volumes."""
    if not properties:
        raise ValueError("No properties found in .g00.gz to map rock types.")

    max_id = max(properties.keys())
    density_lut = np.zeros(max_id + 1, dtype=np.float32)
    susc_lut = np.zeros(max_id + 1, dtype=np.float32)
    for rid, (den, sus) in properties.items():
        density_lut[rid] = den
        susc_lut[rid] = sus

    rock_clipped = np.clip(rock_types, 0, max_id).astype(np.int32, copy=False)
    density = density_lut[rock_clipped]
    susceptibility = susc_lut[rock_clipped]
    return density, susceptibility


def load_observation(
    source: Union[str, io.BufferedIOBase], geom: Dict[str, Any]
) -> np.ndarray:
    """Load mag/grv grid, skipping the 8-line header, and reshape to (ny, nx)."""
    nx, ny = geom["layer_dims"]
    data = load_gz_array(source, skiprows=8)
    expected = nx * ny
    if data.size < expected:
        raise ValueError(
            f"Unexpected obs size {data.size}, expected at least {expected}"
        )
    grid = data[:expected].reshape((ny, nx))
    return grid.astype(np.float32, copy=False)


# ----------------------------
# Discovery helpers
# ----------------------------


def _scan_tar_for_models(tar_path: str) -> List[Dict[str, Any]]:
    """Scan a single tar for complete model file sets; returns locator dicts."""
    locators: List[Dict[str, Any]] = []
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            seen: Dict[str, Dict[str, tarfile.TarInfo]] = {}
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                for ext in (".g12.gz", ".mag.gz", ".grv.gz", ".g00.gz", ".his.gz"):
                    if name.endswith(ext):
                        base = name[: -len(ext)]
                        seen.setdefault(base, {})[ext] = member
            for base, members in seen.items():
                if {".g12.gz", ".mag.gz", ".grv.gz", ".g00.gz"}.issubset(
                    members.keys()
                ):
                    locators.append(
                        {
                            "tar": tar_path,
                            "base": base,
                            "members": members,  # ext -> TarInfo
                        }
                    )
    except tarfile.TarError as exc:
        print(f"Warning: Failed to scan tar {tar_path}: {exc}")
        return []
    return locators


def _count_models_in_tar(tar_path: str) -> int:
    """Count complete models inside a tar without extracting."""
    required_exts = {".g12.gz", ".mag.gz", ".grv.gz", ".g00.gz"}
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            seen: Dict[str, set] = {}
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                for ext in required_exts:
                    if name.endswith(ext):
                        base = name[: -len(ext)]
                        seen.setdefault(base, set()).add(ext)
                        break
        return sum(1 for exts in seen.values() if required_exts.issubset(exts))
    except tarfile.TarError as exc:
        print(f"Warning: Failed to scan tar {tar_path}: {exc}")
        return 0


def _load_first_sample_from_tar(tar_path: str) -> Optional[Dict[str, Any]]:
    """Load the first complete sample from a tar for shape initialization."""
    locators = _scan_tar_for_models(tar_path)
    if not locators:
        return None
    first = locators[0]
    return load_sample_from_tar(first["tar"], first["base"], first["members"])


def _derive_zarr_path(output_dir: str, tar_path: str) -> str:
    """Build a per-tar Zarr store path under the output directory."""
    tar_name = os.path.basename(tar_path)
    if tar_name.endswith(".tar.gz"):
        tar_name = tar_name[: -len(".tar.gz")]
    elif tar_name.endswith(".tar"):
        tar_name = tar_name[: -len(".tar")]
    return os.path.join(output_dir, f"{tar_name}.zarr")


def _is_zarr_complete(store_path: str, expected_count: int) -> Tuple[bool, str]:
    """Return (is_complete, reason) for an existing Zarr store."""
    if not os.path.exists(store_path):
        return False, "missing"
    try:
        store = zarr.open_group(store_path, mode="r")
    except Exception as exc:
        return False, f"open_failed:{type(exc).__name__}"

    for name in ("rock_types", "mag", "grv"):
        if name not in store:
            return False, f"missing_array:{name}"
        try:
            if store[name].shape[0] != expected_count:
                return (
                    False,
                    f"shape_mismatch:{name}:{store[name].shape[0]}",
                )
        except Exception as exc:
            return False, f"shape_failed:{name}:{type(exc).__name__}"

    meta_raw = store.attrs.get("samples_metadata")
    if not meta_raw:
        return False, "missing_metadata"
    try:
        meta = json.loads(meta_raw)
    except Exception:
        return False, "metadata_parse_failed"
    if not isinstance(meta, list):
        return False, "metadata_not_list"
    if len(meta) != expected_count:
        return False, f"metadata_count_mismatch:{len(meta)}"
    return True, "ok"


def _remove_zarr_store(store_path: str) -> None:
    if os.path.isdir(store_path):
        shutil.rmtree(store_path)
    elif os.path.exists(store_path):
        os.remove(store_path)


def _write_samples_to_zarr(
    base_paths: List[str],
    base_offset: int,
    output_path: str,
    read_batch_size: int,
    profile: bool,
) -> Dict[str, Any]:
    """Write samples directly to preallocated Zarr slices."""
    store = zarr.open_group(output_path, mode="r+")
    rock_types = store["rock_types"]
    mag = store["mag"]
    grv = store["grv"]

    metadata_list: List[Dict[str, Any]] = []
    total_load_time = 0.0
    total_write_time = 0.0
    batch_count = 0

    batch_rock: List[np.ndarray] = []
    batch_mag: List[np.ndarray] = []
    batch_grv: List[np.ndarray] = []
    batch_start = base_offset

    for i, base in enumerate(base_paths):
        sample_idx = base_offset + i
        if profile:
            t0 = time.perf_counter()
        try:
            sample = load_sample_files(
                base, profile=profile, sample_idx=sample_idx if profile else None
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sample base '{base}' (index {sample_idx})."
            ) from exc
        if profile:
            total_load_time += time.perf_counter() - t0

        batch_rock.append(sample["rock_types"])
        batch_mag.append(sample["mag"])
        batch_grv.append(sample["grv"])
        metadata_list.append(_serialize_metadata(sample["metadata"]))

        if len(batch_rock) >= read_batch_size or i == len(base_paths) - 1:
            if profile:
                t0 = time.perf_counter()
            n = len(batch_rock)
            batch_end = batch_start + n
            rock_types[batch_start:batch_end] = np.stack(batch_rock, axis=0)
            mag[batch_start:batch_end] = np.stack(batch_mag, axis=0)
            grv[batch_start:batch_end] = np.stack(batch_grv, axis=0)
            if profile:
                total_write_time += time.perf_counter() - t0
            batch_count += 1
            batch_rock = []
            batch_mag = []
            batch_grv = []
            batch_start = batch_end

        del sample

    return {
        "metadata": metadata_list,
        "count": len(metadata_list),
        "load_time": total_load_time,
        "write_time": total_write_time,
        "batches": batch_count,
    }


def _write_samples_chunk_to_zarr(
    args: Tuple[
        int,
        int,
        List[str],
        bool,
        int,
        str,
        str,
        str,
    ],
) -> Dict[str, Any]:
    """Process one chunk of samples and write metadata to disk."""
    (
        chunk_index,
        base_offset,
        base_paths,
        profile,
        read_batch_size,
        output_path,
        tar_name,
        metadata_path,
    ) = args
    try:
        result = _write_samples_to_zarr(
            base_paths,
            base_offset,
            output_path,
            read_batch_size,
            profile,
        )
    except Exception as exc:
        print(
            "Error writing chunk to zarr: "
            f"tar={tar_name} chunk={chunk_index} "
            f"offset={base_offset} count={len(base_paths)} {_format_rss()} "
            f"type={type(exc).__name__} detail={exc!r}"
        )
        traceback.print_exc()
        raise
    with open(metadata_path, "w") as f:
        json.dump(result["metadata"], f)
    if profile:
        print(
            f"Zarr write: tar={tar_name} "
            f"samples={result['count']} "
            f"load={result['load_time']:.2f}s "
            f"write={result['write_time']:.2f}s "
            f"batches={result['batches']}"
        )
    return {
        "chunk_index": chunk_index,
        "metadata_path": metadata_path,
        "count": result["count"],
    }


# ----------------------------
# Filesystem discovery & loading (post-extraction)
# ----------------------------


def find_complete_models(data_root: str) -> List[str]:
    """Return base paths that contain g12, mag, grv, and g00 (his optional)."""
    g12_files = set(
        glob.glob(os.path.join(data_root, "**", "*.g12.gz"), recursive=True)
    )
    mag_files = set(
        glob.glob(os.path.join(data_root, "**", "*.mag.gz"), recursive=True)
    )
    grv_files = set(
        glob.glob(os.path.join(data_root, "**", "*.grv.gz"), recursive=True)
    )
    g00_files = set(
        glob.glob(os.path.join(data_root, "**", "*.g00.gz"), recursive=True)
    )

    bases: List[str] = []
    for g12 in g12_files:
        base = g12.replace(".g12.gz", "")
        if (
            base + ".mag.gz" in mag_files
            and base + ".grv.gz" in grv_files
            and base + ".g00.gz" in g00_files
        ):
            bases.append(base)
    return sorted(bases)


def load_sample_files(
    base_path: str, profile: bool = False, sample_idx: Optional[int] = None
) -> Dict[str, Any]:
    """Load one model from extracted files and return arrays + metadata."""
    g00_path = base_path + ".g00.gz"
    if profile:
        t0 = time.perf_counter()
    props, geom = parse_g00(g00_path)
    if profile:
        t_g00 = time.perf_counter() - t0
        t0 = time.perf_counter()

    rock = load_rock_types(base_path + ".g12.gz", geom, profile=profile)
    if profile:
        t_g12 = time.perf_counter() - t0
        t0 = time.perf_counter()

    mag = load_observation(base_path + ".mag.gz", geom)
    if profile:
        t_mag = time.perf_counter() - t0
        t0 = time.perf_counter()
    grv = load_observation(base_path + ".grv.gz", geom)
    if profile:
        t_grv = time.perf_counter() - t0
        total = t_g00 + t_g12 + t_mag + t_grv
        sample_label = sample_idx if sample_idx is not None else 0
        print(
            f"Sample {sample_label}: g00={t_g00:.2f}s g12={t_g12:.2f}s "
            f"mag={t_mag:.2f}s grv={t_grv:.2f}s total={total:.2f}s"
        )

    meta = {
        "model_id": os.path.basename(base_path),
        "base_path": base_path,
        "g00_path": g00_path,
        "his_path": (
            base_path + ".his.gz" if os.path.exists(base_path + ".his.gz") else None
        ),
        "geometry": geom,
        "properties": props,
    }

    return {
        "rock_types": rock,
        "mag": mag,
        "grv": grv,
        "metadata": meta,
    }


# ----------------------------
# Sample assembly
# ----------------------------


def load_sample_from_tar(
    tar_path: str, base: str, members: Dict[str, tarfile.TarInfo]
) -> Dict[str, Any]:
    """Load one model from an already-scanned tar using provided TarInfo entries."""
    with tarfile.open(tar_path, "r:*") as tf:

        def _open_member(ext: str):
            info = members.get(ext)
            return tf.extractfile(info) if info else None

        g00_member = _open_member(".g00.gz")
        if g00_member is None:
            raise FileNotFoundError(f"Missing g00 in tar {tar_path} for {base}")
        props, geom = parse_g00(g00_member)

        rock = load_rock_types(_open_member(".g12.gz"), geom)

        mag = load_observation(_open_member(".mag.gz"), geom)
        grv = load_observation(_open_member(".grv.gz"), geom)

        meta = {
            "model_id": os.path.basename(base),
            "base_path": base,
            "tar_path": tar_path,
            "g00_member": base + ".g00.gz",
            "his_member": base + ".his.gz" if members.get(".his.gz") else None,
            "geometry": geom,
            "properties": props,
        }

    return {
        "rock_types": rock,
        "mag": mag,
        "grv": grv,
        "metadata": meta,
    }


def _serialize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata to JSON-serializable format (tuples and dicts with tuple values)."""
    result = {}
    for key, value in meta.items():
        if key == "geometry":
            # Geometry has tuple values
            result[key] = {
                k: list(v) if isinstance(v, tuple) else v for k, v in value.items()
            }
        elif key == "properties":
            # Properties is dict of int -> (float, float) tuples
            result[key] = {str(k): list(v) for k, v in value.items()}
        else:
            result[key] = value
    return result


def process_tar_to_zarr(
    tar_path: str,
    temp_root: str,
    expected_count: int,
    profile: bool,
    max_samples: Optional[int],
    read_batch_size: int,
    num_workers_per_tar: int,
    output_dir: str,
    chunk_3d: Tuple[int, int, int, int],
    chunk_2d: Tuple[int, int, int],
) -> Dict[str, Any]:
    """Process one tar and write its samples into a per-tar Zarr store."""
    os.makedirs(temp_root, exist_ok=True)
    temp_dir = os.path.join(
        temp_root, os.path.basename(tar_path).replace(".", "_") + f"_{uuid.uuid4().hex}"
    )
    os.makedirs(temp_dir, exist_ok=True)

    stage = "init"
    try:
        print(f"Processing tar: {tar_path} {_format_rss()}")
        stage = "extract"
        t0 = time.perf_counter()
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(path=temp_dir)
        if profile:
            print(f"Tar extraction: {time.perf_counter() - t0:.2f}s {_format_rss()}")

        stage = "discover"
        t0 = time.perf_counter()
        base_paths = find_complete_models(temp_dir)
        if profile:
            print(
                f"File discovery: {time.perf_counter() - t0:.2f}s "
                f"({len(base_paths)} models found) {_format_rss()}"
            )

        if max_samples is not None:
            base_paths = base_paths[:max_samples]

        count = len(base_paths)
        if count == 0:
            if expected_count != 0:
                raise RuntimeError(
                    f"Expected {expected_count} samples from {tar_path}, found 0."
                )
            return {"count": 0, "metadata": [], "store_path": None}

        if count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} samples from {tar_path}, found {count}."
            )

        tar_name = os.path.basename(tar_path)
        store_path = _derive_zarr_path(output_dir, tar_path)
        stage = "init_zarr"
        first_sample = load_sample_files(base_paths[0], profile=False)
        writer = ZarrStreamWriter(store_path, chunk_3d, chunk_2d)
        writer.init_arrays_for_total(first_sample, count)
        del first_sample

        start_idx = 0
        stage = "write_samples"
        if num_workers_per_tar and num_workers_per_tar > 1:
            num_workers = min(num_workers_per_tar, count)
            chunk_size = int(math.ceil(count / num_workers))
            if chunk_size % read_batch_size != 0:
                chunk_size = int(
                    math.ceil(chunk_size / read_batch_size) * read_batch_size
                )
            work_items = []
            for chunk_index, start in enumerate(range(0, count, chunk_size)):
                end = min(start + chunk_size, count)
                chunk_bases = base_paths[start:end]
                chunk_metadata_path = os.path.join(
                    temp_dir, f"metadata.part{chunk_index}.json"
                )
                work_items.append(
                    (
                        chunk_index,
                        start_idx + start,
                        chunk_bases,
                        profile,
                        read_batch_size,
                        store_path,
                        tar_name,
                        chunk_metadata_path,
                    )
                )

            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for result in executor.map(_write_samples_chunk_to_zarr, work_items):
                    results.append(result)

            results.sort(key=lambda r: r["chunk_index"])
            metadata_list: List[Dict[str, Any]] = []
            for result in results:
                with open(result["metadata_path"], "r") as f:
                    metadata_list.extend(json.load(f))
                os.remove(result["metadata_path"])
        else:
            result = _write_samples_to_zarr(
                base_paths,
                start_idx,
                store_path,
                read_batch_size,
                profile,
            )
            metadata_list = result["metadata"]
            if profile:
                print(
                    f"Zarr write: tar={tar_name} "
                    f"samples={result['count']} "
                    f"load={result['load_time']:.2f}s "
                    f"write={result['write_time']:.2f}s "
                    f"batches={result['batches']}"
                )
        writer.metadata = metadata_list
        writer.sample_idx = count
        stage = "finalize"
        writer.finalize()
        print(f"Processed tar: {tar_path} {_format_rss()} (num samples: {count})")
        return {"count": count, "metadata": metadata_list, "store_path": store_path}
    except Exception as e:
        print(
            "Error processing tar to zarr: "
            f"tar={tar_path} stage={stage} {_format_rss()} "
            f"type={type(e).__name__} detail={e!r}"
        )
        traceback.print_exc()
        raise
    finally:
        print(f"Removing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# Zarr writer
# ----------------------------


class ZarrStreamWriter:
    """Writer that initializes Zarr arrays and stores metadata."""

    def __init__(
        self,
        output_path: str,
        chunk_3d: Tuple[int, int, int, int],
        chunk_2d: Tuple[int, int, int],
    ):
        self.output_path = output_path
        self.chunk_3d = chunk_3d
        self.chunk_2d = chunk_2d
        self.store: Optional[zarr.Group] = None
        self.sample_idx = 0
        self.metadata: List[Dict[str, Any]] = []

    def init_arrays_for_total(self, sample: Dict[str, Any], total_samples: int) -> None:
        """Initialize fixed-size Zarr arrays based on first sample's shape."""
        self.store = zarr.open_group(self.output_path, mode="w")
        expected_3d_shape = sample["rock_types"].shape
        expected_2d_shape = sample["mag"].shape

        shuffle = getattr(Blosc, "BITSHUFFLE", 2)
        compressors = [Blosc(cname="zstd", clevel=3, shuffle=shuffle)]

        # Create fixed-size arrays (no resize)
        nz, nx, ny = expected_3d_shape
        mx, my = expected_2d_shape

        self.store.create_array(
            "rock_types",
            shape=(total_samples, nz, nx, ny),
            chunks=self.chunk_3d,
            dtype=np.uint8,
            compressors=compressors,
        )
        self.store.create_array(
            "mag",
            shape=(total_samples, mx, my),
            chunks=self.chunk_2d,
            dtype=np.float32,
            compressors=compressors,
        )
        self.store.create_array(
            "grv",
            shape=(total_samples, mx, my),
            chunks=self.chunk_2d,
            dtype=np.float32,
            compressors=compressors,
        )

    def finalize(self) -> None:
        """Finalize the Zarr store by writing metadata."""
        if self.store is not None:
            self.store.attrs["samples_metadata"] = json.dumps(self.metadata)

    @property
    def count(self) -> int:
        """Return number of samples written."""
        return self.sample_idx


# ----------------------------
# CLI
# ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Noddyverse models into per-tar Zarr stores."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing Noddyverse tar archives.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for per-tar Zarr stores.",
    )
    parser.add_argument(
        "--chunk-3d",
        type=int,
        nargs=4,
        default=(16, 64, 64, 64),
        metavar=("S", "Z", "X", "Y"),
        help="Chunks for 3D arrays (sample, z, x, y).",
    )
    parser.add_argument(
        "--chunk-2d",
        type=int,
        nargs=3,
        default=(16, 128, 128),
        metavar=("S", "X", "Y"),
        help="Chunks for 2D arrays (sample, x, y).",
    )
    parser.add_argument(
        "--verify-orientation",
        action="store_true",
        help="Print quick stats on first sample to confirm shapes/orientation.",
    )
    parser.add_argument(
        "--num-parallel-tars",
        type=int,
        default=1,
        help="Number of tar archives to process in parallel (1 for serial).",
    )
    parser.add_argument(
        "--num-workers-per-tar",
        type=int,
        default=1,
        help="Workers to load samples within each tar (1 for serial).",
    )
    parser.add_argument(
        "--temp-root",
        type=str,
        default=None,
        help="Directory to place per-tar temporary extraction folders (defaults to data-root).",
        required=True,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling: print per-sample timing breakdown.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process per tar (for quick profiling runs).",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=16,
        help="Batch size for reading temp binaries into Zarr writes.",
    )

    args = parser.parse_args()
    temp_root = args.temp_root
    os.makedirs(args.output, exist_ok=True)

    tar_paths = glob.glob(os.path.join(args.data_root, "*.tar")) + glob.glob(
        os.path.join(args.data_root, "*.tar.gz")
    )
    if not tar_paths:
        raise ValueError(f"No tar files found under {args.data_root}")

    with ProcessPoolExecutor(max_workers=32) as executor:
        counts = list(executor.map(_count_models_in_tar, tar_paths))

    tar_jobs = []
    for tar_path, count in zip(tar_paths, counts):
        if args.max_samples is not None:
            count = min(count, args.max_samples)
        if count == 0:
            continue
        store_path = _derive_zarr_path(args.output, tar_path)
        if os.path.exists(store_path):
            is_complete, reason = _is_zarr_complete(store_path, count)
            if is_complete:
                print(
                    f"Skipping tar {tar_path}: existing zarr complete "
                    f"({count} samples)."
                )
                continue
            print(
                f"Removing incomplete zarr {store_path} "
                f"(reason={reason}) before reprocessing tar {tar_path}."
            )
            _remove_zarr_store(store_path)
        tar_jobs.append({"tar_path": tar_path, "count": count})

    if not tar_jobs:
        raise ValueError(
            f"No complete models found under extracted data from {args.data_root}"
        )

    if args.num_parallel_tars and args.num_parallel_tars > 1:
        with ProcessPoolExecutor(max_workers=args.num_parallel_tars) as executor:
            future_to_job = {
                executor.submit(
                    process_tar_to_zarr,
                    job["tar_path"],
                    temp_root,
                    job["count"],
                    args.profile,
                    args.max_samples,
                    args.read_batch_size,
                    args.num_workers_per_tar,
                    args.output,
                    tuple(args.chunk_3d),
                    tuple(args.chunk_2d),
                ): job
                for job in tar_jobs
            }
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    fut.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to process tar {job['tar_path']}: {exc}"
                    ) from exc
    else:
        for job in tar_jobs:
            process_tar_to_zarr(
                job["tar_path"],
                temp_root,
                job["count"],
                args.profile,
                args.max_samples,
                args.read_batch_size,
                args.num_workers_per_tar,
                args.output,
                tuple(args.chunk_3d),
                tuple(args.chunk_2d),
            )

    if args.verify_orientation:
        first_sample = _load_first_sample_from_tar(tar_jobs[0]["tar_path"])
        if first_sample is None:
            raise ValueError("Failed to load a sample for orientation verification.")
        print(
            f"First sample {first_sample['metadata']['model_id']}: "
            f"rock_types shape {first_sample['rock_types'].shape}, "
            f"mag shape {first_sample['mag'].shape}, "
            f"grv shape {first_sample['grv'].shape}, "
            f"rock min/max {first_sample['rock_types'].min()}/{first_sample['rock_types'].max()}"
        )
        print(f"Saved per-tar Zarr stores under {args.output}")


if __name__ == "__main__":
    main()
