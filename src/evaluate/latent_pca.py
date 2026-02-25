"""
PCA analysis on VAE latents or raw density/susceptibility data stored in Zarr format.

This script:
- Auto-detects input type (latents vs raw data) from Zarr store structure.
- Streams data from a Zarr store (no normalization).
- Uses PCA (CPU or GPU-accelerated) to fit on large datasets.
- Saves:
  - Explained variance ratios.
  - Scree plot.
  - 2D scatter plots for a few principal component pairs.
  - Optional histograms of individual principal components.

Input types:
- Latents: Zarr store with 'latent_mu' array (VAE-encoded latents).
- Raw data: Zarr store with 'rock_types' array (200^3 density + susceptibility volumes).

Example usage (YAML config only):

    python -m src.evaluate.latent_pca \\
        --config config/evaluate_latent_pca_raw_susc.yaml
"""

from __future__ import annotations

# Set OpenBLAS threading BEFORE importing numpy/sklearn
# This must happen before any BLAS/LAPACK libraries are loaded
import os

if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = str(min(128, os.cpu_count() or 1))
    print(f"Set OPENBLAS_NUM_THREADS to {os.environ['OPENBLAS_NUM_THREADS']}")
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = str(min(128, os.cpu_count() or 1))
    print(f"Set MKL_NUM_THREADS to {os.environ['MKL_NUM_THREADS']}")
if "NUMEXPR_NUM_THREADS" not in os.environ:
    os.environ["NUMEXPR_NUM_THREADS"] = str(min(128, os.cpu_count() or 1))
    print(f"Set NUMEXPR_NUM_THREADS to {os.environ['NUMEXPR_NUM_THREADS']}")


import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import zarr
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.filtering import load_filter_stats_cache, select_indices_from_filter_stats
from ..utils.datasets import JointDensitySuscDataset, LatentZarrDataset

logger = logging.getLogger(__name__)

# Optional GPU imports (cuML/RAPIDS)
CUPY_AVAILABLE = False
CUML_AVAILABLE = False
DASK_CUDA_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None

try:
    from cuml.decomposition import IncrementalPCA as cuIPCA

    CUML_AVAILABLE = True
except ImportError:
    cuIPCA = None

try:
    from dask.distributed import Client, get_worker, wait
    from dask_cuda import LocalCUDACluster

    DASK_CUDA_AVAILABLE = True
except ImportError:
    LocalCUDACluster = None
    Client = None
    get_worker = None
    wait = None


RAW_GPU_WORKER_STATE: Dict[str, Any] = {}


def _get_raw_gpu_worker_state() -> Optional[Dict[str, Any]]:
    """Get worker-local raw PCA state, preferring Dask worker storage."""
    if get_worker is None:
        return RAW_GPU_WORKER_STATE if RAW_GPU_WORKER_STATE else None
    try:
        worker = get_worker()
    except Exception:
        return RAW_GPU_WORKER_STATE if RAW_GPU_WORKER_STATE else None
    state = getattr(worker, "raw_gpu_worker_state", None)
    if state:
        return state
    return RAW_GPU_WORKER_STATE if RAW_GPU_WORKER_STATE else None


def _set_raw_gpu_worker_state(state: Dict[str, Any]) -> None:
    """Set worker-local raw PCA state on Dask worker when available."""
    global RAW_GPU_WORKER_STATE
    RAW_GPU_WORKER_STATE = state
    if get_worker is None:
        return
    try:
        worker = get_worker()
        worker.raw_gpu_worker_state = state
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental PCA over VAE latents or raw data."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config path for latent_pca settings.",
    )
    cli_args = parser.parse_args()

    with Path(cli_args.config).open("r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping/dict: {cli_args.config}")

    defaults: Dict[str, Any] = {
        "batch_size": 8,
        "num_workers": 4,
        "max_samples": None,
        "sample_stride": 1,
        "n_components": 10,
        "scatter_max_points": 5000,
        "seed": 0,
        "raw_downsample_size": 64,
        "raw_field": "both",
        "raw_projection_dim": 4096,
        "use_filter_cache": False,
        "filter_cache_path": None,
        "susc_active_threshold_log10": -2.5,
        "min_active_frac": 0.005,
        "low_info_keep_prob": 0.2,
        "use_gpu": False,
        "num_gpus": 1,
        "gpu_id": 0,
    }
    required = {"latents_zarr", "output_dir"}

    config_data = {**defaults, **loaded}
    missing = sorted(key for key in required if config_data.get(key) is None)
    if missing:
        raise ValueError(
            f"Missing required config keys in {cli_args.config}: {', '.join(missing)}"
        )
    if config_data["raw_field"] not in {"both", "density", "susc"}:
        raise ValueError("raw_field must be one of: both, density, susc")

    config_data["config"] = cli_args.config
    return argparse.Namespace(**config_data)


def _detect_store_mode(zarr_path: str) -> str:
    """Detect whether a Zarr store contains latents or raw data."""
    group = zarr.open_group(zarr_path, mode="r")
    is_latent_store = "latent_mu" in group
    is_raw_store = "rock_types" in group and "latent_mu" not in group
    if is_latent_store:
        return "latent"
    if is_raw_store:
        return "raw"
    raise ValueError(
        f"Zarr store at {zarr_path} does not appear to be a valid latent or raw data store. "
        f"Expected either 'latent_mu' (for latents) or 'rock_types' (for raw data)."
    )


def _build_dataloader(
    zarr_path: str,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    sample_stride: int,
    raw_field: str,
    selected_indices: Optional[list[int]] = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, str]:
    """
    Build a DataLoader for PCA analysis.

    Auto-detects whether the Zarr store contains:
    - Latents: has 'latent_mu' array -> uses LatentZarrDataset
    - Raw data: has 'rock_types' array but no 'latent_mu' -> uses JointDensitySuscDataset

    Returns:
        Tuple of (DataLoader, mode_string) where mode_string is "latent" or "raw"
    """
    mode = _detect_store_mode(zarr_path)
    if mode == "latent":
        dataset = LatentZarrDataset(zarr_path, pad_to=32, use_logvar=True)
    elif mode == "raw":
        dataset = JointDensitySuscDataset(
            zarr_path, return_index=False, field=raw_field
        )
    else:
        raise AssertionError(f"Unexpected mode: {mode}")

    if selected_indices is None:
        indices = list(range(0, len(dataset), 1))
    else:
        indices = [int(i) for i in selected_indices]
        bad = [i for i in indices if i < 0 or i >= len(dataset)]
        if bad:
            raise ValueError(
                f"Selected index out of bounds for dataset of size {len(dataset)}; "
                f"first invalid index: {bad[0]}"
            )

    stride = max(sample_stride, 1)
    indices = indices[::stride]
    if max_samples is not None:
        indices = indices[:max_samples]

    subset = torch.utils.data.Subset(dataset, indices)

    loader_kwargs = dict(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)

    return DataLoader(**loader_kwargs), mode


def _flatten_batch(batch: torch.Tensor) -> np.ndarray:
    # batch: (B, C, D, H, W) -> (B, C*D*H*W)
    if batch.ndim != 5:
        raise ValueError(
            f"Expected latents with 5 dims (B,C,D,H,W), got shape {tuple(batch.shape)}"
        )
    b = batch.size(0)
    return batch.view(b, -1).cpu().numpy()


def _flatten_batch_cupy(batch: torch.Tensor, device_id: int = 0):
    """Flatten batch and return as CuPy array on specified GPU."""
    if batch.ndim != 5:
        raise ValueError(
            f"Expected latents with 5 dims (B,C,D,H,W), got shape {tuple(batch.shape)}"
        )
    b = batch.size(0)
    flat_np = batch.view(b, -1).cpu().numpy()
    with cp.cuda.Device(device_id):
        return cp.asarray(flat_np)


def _downsample_and_flatten_raw_batch(
    batch: torch.Tensor, downsample_size: int
) -> np.ndarray:
    """Downsample raw 3D data and flatten to (B, features) on CPU."""
    if batch.ndim != 5:
        raise ValueError(
            f"Expected latents with 5 dims (B,C,D,H,W), got shape {tuple(batch.shape)}"
        )
    if downsample_size < 1:
        raise ValueError(f"raw_downsample_size must be >= 1, got {downsample_size}")

    batch = batch.to(dtype=torch.float32)
    target_size = (downsample_size, downsample_size, downsample_size)
    if tuple(batch.shape[2:]) != target_size:
        batch = torch.nn.functional.interpolate(
            batch, size=target_size, mode="trilinear", align_corners=False
        )

    b = batch.size(0)
    return batch.view(b, -1).cpu().numpy()


def _batch_to_numpy(batch: Any) -> np.ndarray:
    """Convert a dataloader batch to a numpy array."""
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    if isinstance(batch, (list, tuple)) and batch:
        # Support datasets that may return (idx, tensor)
        candidate = batch[-1]
        if isinstance(candidate, torch.Tensor):
            return candidate.detach().cpu().numpy()
    raise TypeError(f"Unsupported batch type for raw PCA: {type(batch)!r}")


def _project_raw_batch(
    projector: GaussianRandomProjection, flat_batch: np.ndarray
) -> np.ndarray:
    """Project flattened raw features to a lower-dimensional space."""
    return projector.transform(flat_batch)


def _raw_gpu_project_from_raw_shard(
    raw_shard: np.ndarray,
    feature_indices: np.ndarray,
    downsample_size: int,
) -> torch.Tensor:
    """Downsample and project a raw shard on the worker GPU."""
    if raw_shard.size == 0:
        return torch.empty(
            (0, int(feature_indices.shape[0])), device="cuda", dtype=torch.float32
        )

    x = torch.as_tensor(raw_shard, dtype=torch.float32, device="cuda")
    if x.ndim != 5:
        raise ValueError(
            f"Expected raw shard with 5 dims (B,C,D,H,W), got shape {tuple(x.shape)}"
        )
    target_size = (downsample_size, downsample_size, downsample_size)
    if tuple(x.shape[2:]) != target_size:
        x = torch.nn.functional.interpolate(
            x, size=target_size, mode="trilinear", align_corners=False
        )
    x = x.reshape(x.shape[0], -1)
    feat_idx = torch.as_tensor(feature_indices, device=x.device, dtype=torch.long)
    return x.index_select(1, feat_idx)


def _raw_gpu_batch_stats_from_flat(
    flat_shard: np.ndarray,
    feature_indices: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Compute projected-batch sufficient statistics on a GPU worker."""
    if flat_shard.size == 0:
        k = int(feature_indices.shape[0])
        return 0, np.zeros((k,), dtype=np.float64), np.zeros((k, k), dtype=np.float64)

    x = cp.asarray(flat_shard, dtype=cp.float32)
    feat_idx = cp.asarray(feature_indices)
    y = x[:, feat_idx]

    n_local = int(y.shape[0])
    sum_local = cp.asnumpy(cp.sum(y, axis=0, dtype=cp.float64))
    gram_local = cp.asnumpy(cp.matmul(y.T, y)).astype(np.float64, copy=False)

    del x, feat_idx, y
    return n_local, sum_local, gram_local


def _raw_gpu_batch_transform_from_flat(
    flat_shard: np.ndarray,
    feature_indices: np.ndarray,
    mean_proj: np.ndarray,
    components_proj: np.ndarray,
) -> np.ndarray:
    """Transform projected shard into PCA scores on a GPU worker."""
    if flat_shard.size == 0:
        return np.empty((0, components_proj.shape[1]), dtype=np.float32)

    x = cp.asarray(flat_shard, dtype=cp.float32)
    feat_idx = cp.asarray(feature_indices)
    mean_cp = cp.asarray(mean_proj, dtype=cp.float32)
    comps_cp = cp.asarray(components_proj, dtype=cp.float32)

    y = x[:, feat_idx]
    centered = y - mean_cp
    scores = cp.matmul(centered, comps_cp)
    scores_np = cp.asnumpy(scores)

    del x, feat_idx, mean_cp, comps_cp, y, centered, scores
    return scores_np


def _raw_gpu_worker_reset_state(
    feature_indices: np.ndarray,
    downsample_size: int,
) -> bool:
    """Initialize per-worker accumulators for raw multi-GPU PCA."""
    k = int(feature_indices.shape[0])
    state = {
        "feature_indices": np.asarray(feature_indices, dtype=np.int64),
        "downsample_size": int(downsample_size),
        "n": 0,
        "sum": np.zeros((k,), dtype=np.float64),
        "gram": np.zeros((k, k), dtype=np.float64),
    }
    _set_raw_gpu_worker_state(state)
    return True


def _raw_gpu_worker_accumulate_shard(
    raw_shard: np.ndarray,
    feature_indices: Optional[np.ndarray] = None,
    downsample_size: Optional[int] = None,
) -> int:
    """Accumulate covariance statistics for one shard on a GPU worker."""
    state = _get_raw_gpu_worker_state()
    if not state:
        if feature_indices is None or downsample_size is None:
            raise RuntimeError(
                "RAW_GPU_WORKER_STATE not initialized on worker and no initialization "
                "inputs were provided to this task."
            )
        _raw_gpu_worker_reset_state(
            feature_indices=np.asarray(feature_indices, dtype=np.int64),
            downsample_size=int(downsample_size),
        )
        state = _get_raw_gpu_worker_state()
        if not state:
            raise RuntimeError("RAW_GPU_WORKER_STATE could not be initialized on worker.")
    if raw_shard.size == 0:
        return 0

    feature_indices = state["feature_indices"]
    downsample_size = int(state["downsample_size"])

    y = _raw_gpu_project_from_raw_shard(
        raw_shard=raw_shard,
        feature_indices=feature_indices,
        downsample_size=downsample_size,
    )
    n_local = int(y.shape[0])
    sum_local = y.sum(dim=0, dtype=torch.float64).cpu().numpy()
    gram_local = torch.matmul(y.T, y).to(dtype=torch.float64).cpu().numpy()

    state["n"] += n_local
    state["sum"] += sum_local
    state["gram"] += gram_local
    return n_local


def _raw_gpu_worker_get_state(
    expected_feature_dim: Optional[int] = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return accumulated sufficient statistics from a GPU worker."""
    state = _get_raw_gpu_worker_state()
    if not state:
        if expected_feature_dim is None:
            raise RuntimeError("RAW_GPU_WORKER_STATE not initialized on worker.")
        k = int(expected_feature_dim)
        return 0, np.zeros((k,), dtype=np.float64), np.zeros((k, k), dtype=np.float64)
    return (
        int(state["n"]),
        state["sum"],
        state["gram"],
    )


def _raw_gpu_worker_transform_shard(
    raw_shard: np.ndarray,
    feature_indices: np.ndarray,
    downsample_size: int,
    mean_proj: np.ndarray,
    components_proj: np.ndarray,
) -> np.ndarray:
    """Transform one raw shard to PCA scores on a GPU worker."""
    if raw_shard.size == 0:
        return np.empty((0, int(components_proj.shape[1])), dtype=np.float32)

    y = _raw_gpu_project_from_raw_shard(
        raw_shard=raw_shard,
        feature_indices=feature_indices,
        downsample_size=downsample_size,
    )
    mean_t = torch.as_tensor(mean_proj, device=y.device, dtype=torch.float32)
    comps_t = torch.as_tensor(components_proj, device=y.device, dtype=torch.float32)
    scores = torch.matmul(y - mean_t, comps_t)
    return scores.cpu().numpy()


def _wait_for_requested_workers(
    client: Client,
    expected_workers: int,
    timeout_seconds: int = 90,
    poll_interval_seconds: float = 1.0,
    stable_polls: int = 3,
) -> list[str]:
    """Wait until the requested number of Dask workers is available and stable."""
    if expected_workers < 1:
        raise ValueError(f"expected_workers must be >= 1, got {expected_workers}")
    stable_polls = max(1, int(stable_polls))

    start = time.time()
    last_count = -1
    consecutive_stable = 0

    while True:
        info = client.scheduler_info()
        workers_info = info.get("workers", {})
        workers = sorted(workers_info.keys())
        count = len(workers)

        if count != last_count:
            logger.info(
                "Dask worker availability: %d/%d ready.",
                count,
                expected_workers,
            )
            last_count = count
            consecutive_stable = 0
        else:
            consecutive_stable += 1

        if count >= expected_workers and consecutive_stable >= stable_polls:
            selected = workers[:expected_workers]
            logger.info("Using workers: %s", selected)
            return selected

        if time.time() - start > timeout_seconds:
            summaries = []
            for addr in workers:
                w = workers_info.get(addr, {})
                summaries.append(
                    f"{addr}(name={w.get('name', 'unknown')}, nthreads={w.get('nthreads', 'unknown')})"
                )
            raise RuntimeError(
                "Timed out waiting for "
                f"{expected_workers} workers; only {count} available. "
                f"Current workers: {summaries}"
            )
        time.sleep(max(0.1, float(poll_interval_seconds)))


def _plot_scree(explained_variance_ratio: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    x = np.arange(1, len(explained_variance_ratio) + 1)
    plt.plot(x, explained_variance_ratio, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA Scree Plot")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_scatter(
    components: np.ndarray,
    i: int,
    j: int,
    output_path: Path,
    max_points: int,
    seed: int,
) -> None:
    if components.shape[1] <= max(i, j):
        logger.warning(
            "Not enough components for scatter plot PC%d vs PC%d (have %d).",
            i + 1,
            j + 1,
            components.shape[1],
        )
        return

    rng = np.random.default_rng(seed)
    x = components[:, i]
    y = components[:, j]
    n = x.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=3, alpha=0.6)
    plt.xlabel(f"PC{i+1}")
    plt.ylabel(f"PC{j+1}")
    plt.title(f"PCA Scatter: PC{i+1} vs PC{j+1}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_pc_histogram(
    components: np.ndarray,
    pc_index: int,
    output_path: Path,
    bins: int = 50,
) -> None:
    if components.shape[1] <= pc_index:
        logger.warning(
            "Not enough components for histogram of PC%d (have %d).",
            pc_index + 1,
            components.shape[1],
        )
        return

    data = components[:, pc_index]
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, alpha=0.8)
    plt.xlabel(f"PC{pc_index+1} score")
    plt.ylabel("Count")
    plt.title(f"Histogram of PC{pc_index+1}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _run_pca_cpu(
    loader: DataLoader,
    n_components: int,
    batch_size: int,
) -> tuple:
    """Run IncrementalPCA on CPU using sklearn."""
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # First pass: partial_fit
    total_samples = 0
    with tqdm(loader, desc="Fitting PCA (CPU)", unit="batch") as pbar:
        for batch in pbar:
            flat = _flatten_batch(batch)
            ipca.partial_fit(flat)
            total_samples += flat.shape[0]
            pbar.set_postfix({"samples": total_samples})
    logger.info("IncrementalPCA partial_fit complete over %d samples.", total_samples)

    # Second pass: transform to collect components for plotting
    logger.info("Running IncrementalPCA.transform to collect component scores.")
    components_list = []
    with tqdm(loader, desc="Transforming (CPU)", unit="batch") as pbar:
        for batch in pbar:
            flat = _flatten_batch(batch)
            transformed = ipca.transform(flat)
            components_list.append(transformed)
            pbar.set_postfix({"samples": sum(c.shape[0] for c in components_list)})

    components = np.concatenate(components_list, axis=0)
    return ipca, components


def _run_pca_raw_approx_cpu(
    loader: DataLoader,
    n_components: int,
    batch_size: int,
    raw_downsample_size: int,
    raw_projection_dim: int,
    seed: int,
) -> tuple:
    """Run approximate PCA for raw data: downsample -> random projection -> IncrementalPCA."""
    if raw_downsample_size < 1:
        raise ValueError(f"raw_downsample_size must be >= 1, got {raw_downsample_size}")
    if raw_projection_dim < n_components:
        raise ValueError(
            f"raw_projection_dim ({raw_projection_dim}) must be >= n_components ({n_components})."
        )

    projector = GaussianRandomProjection(
        n_components=raw_projection_dim,
        random_state=seed,
    )
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # First pass: fit projection once, then partial_fit PCA in projected space
    total_samples = 0
    projector_fitted = False
    with tqdm(loader, desc="Fitting PCA (raw approx, CPU)", unit="batch") as pbar:
        for batch in pbar:
            flat = _downsample_and_flatten_raw_batch(batch, raw_downsample_size)
            if not projector_fitted:
                projector.fit(flat)
                projector_fitted = True
            projected = _project_raw_batch(projector, flat)
            ipca.partial_fit(projected)
            total_samples += projected.shape[0]
            pbar.set_postfix({"samples": total_samples})

    logger.info(
        "Raw approximate IncrementalPCA partial_fit complete over %d samples.",
        total_samples,
    )

    # Second pass: transform to collect component scores
    logger.info("Running raw approximate IncrementalPCA.transform.")
    components_list = []
    with tqdm(loader, desc="Transforming (raw approx, CPU)", unit="batch") as pbar:
        for batch in pbar:
            flat = _downsample_and_flatten_raw_batch(batch, raw_downsample_size)
            projected = _project_raw_batch(projector, flat)
            transformed = ipca.transform(projected)
            components_list.append(transformed)
            pbar.set_postfix({"samples": sum(c.shape[0] for c in components_list)})

    components = np.concatenate(components_list, axis=0)
    return ipca, components


def _build_pca_result(
    explained_variance_ratio: np.ndarray,
    singular_values: np.ndarray,
    n_components: int,
    total_samples: int,
):
    """Create a lightweight sklearn-like PCA result object."""

    class PCAResult:
        pass

    result = PCAResult()
    result.explained_variance_ratio_ = explained_variance_ratio
    result.singular_values_ = singular_values
    result.n_components_ = n_components
    result.n_samples_seen_ = total_samples
    return result


def _compute_pca_from_projected_covariance(
    gram_total: np.ndarray,
    sum_total: np.ndarray,
    total_samples: int,
    n_components: int,
    device_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal directions in projected space from covariance stats."""
    if total_samples < 2:
        raise ValueError(f"Need at least 2 samples for PCA, got {total_samples}.")

    mean_proj = sum_total / float(total_samples)
    cov = (gram_total - total_samples * np.outer(mean_proj, mean_proj)) / float(
        total_samples - 1
    )

    with cp.cuda.Device(device_id):
        cov_cp = cp.asarray(cov, dtype=cp.float32)
        evals_cp, evecs_cp = cp.linalg.eigh(cov_cp)
        evals = cp.asnumpy(evals_cp)
        evecs = cp.asnumpy(evecs_cp)

    order = np.argsort(evals)[::-1]
    evals_sorted = np.maximum(evals[order], 0.0)
    components_proj = evecs[:, order[:n_components]].astype(np.float32, copy=False)
    top_evals = evals_sorted[:n_components]

    denom = float(np.sum(evals_sorted))
    if denom > 0.0:
        explained_variance_ratio = top_evals / denom
    else:
        explained_variance_ratio = np.zeros_like(top_evals)

    singular_values = np.sqrt(top_evals * float(total_samples - 1))
    return (
        mean_proj.astype(np.float32, copy=False),
        components_proj,
        explained_variance_ratio.astype(np.float64, copy=False),
        singular_values.astype(np.float64, copy=False),
    )


def _run_pca_raw_approx_single_gpu(
    loader: DataLoader,
    n_components: int,
    raw_downsample_size: int,
    raw_projection_dim: int,
    seed: int,
    gpu_id: int = 0,
) -> tuple:
    """Run approximate PCA on raw data using one GPU."""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available but raw GPU mode was requested.")
    if raw_downsample_size < 1:
        raise ValueError(f"raw_downsample_size must be >= 1, got {raw_downsample_size}")
    if raw_projection_dim < n_components:
        raise ValueError(
            f"raw_projection_dim ({raw_projection_dim}) must be >= n_components ({n_components})."
        )

    rng = np.random.default_rng(seed)
    feature_indices: Optional[np.ndarray] = None
    total_samples = 0
    sum_total: Optional[np.ndarray] = None
    gram_total: Optional[np.ndarray] = None

    logger.info(
        "Running raw approximate PCA on GPU:%d with random feature subsampling projection_dim=%d.",
        gpu_id,
        raw_projection_dim,
    )
    with tqdm(
        loader, desc=f"Fitting PCA (raw approx, GPU:{gpu_id})", unit="batch"
    ) as pbar:
        for batch in pbar:
            flat = _downsample_and_flatten_raw_batch(batch, raw_downsample_size)

            if feature_indices is None:
                n_features = int(flat.shape[1])
                if raw_projection_dim > n_features:
                    raise ValueError(
                        f"raw_projection_dim ({raw_projection_dim}) must be <= projected feature count ({n_features})."
                    )
                feature_indices = np.sort(
                    rng.choice(n_features, size=raw_projection_dim, replace=False)
                ).astype(np.int64, copy=False)
                sum_total = np.zeros((raw_projection_dim,), dtype=np.float64)
                gram_total = np.zeros(
                    (raw_projection_dim, raw_projection_dim), dtype=np.float64
                )

            n_local, sum_local, gram_local = _raw_gpu_batch_stats_from_flat(
                flat, feature_indices
            )
            total_samples += n_local
            sum_total += sum_local
            gram_total += gram_local
            pbar.set_postfix({"samples": total_samples})

    if feature_indices is None or sum_total is None or gram_total is None:
        raise ValueError("No raw data batches were available for PCA.")

    mean_proj, components_proj, explained_variance_ratio, singular_values = (
        _compute_pca_from_projected_covariance(
            gram_total=gram_total,
            sum_total=sum_total,
            total_samples=total_samples,
            n_components=n_components,
            device_id=gpu_id,
        )
    )

    logger.info("Transforming raw data on GPU:%d to collect component scores.", gpu_id)
    components_list = []
    with tqdm(
        loader, desc=f"Transforming (raw approx, GPU:{gpu_id})", unit="batch"
    ) as pbar:
        for batch in pbar:
            flat = _downsample_and_flatten_raw_batch(batch, raw_downsample_size)
            scores = _raw_gpu_batch_transform_from_flat(
                flat, feature_indices, mean_proj, components_proj
            )
            components_list.append(scores)
            pbar.set_postfix({"samples": sum(c.shape[0] for c in components_list)})

    components = np.concatenate(components_list, axis=0)
    result = _build_pca_result(
        explained_variance_ratio=explained_variance_ratio,
        singular_values=singular_values,
        n_components=n_components,
        total_samples=total_samples,
    )
    return result, components


def _run_pca_raw_approx_multi_gpu(
    loader: DataLoader,
    n_components: int,
    raw_downsample_size: int,
    raw_projection_dim: int,
    seed: int,
    num_gpus: int,
) -> tuple:
    """Run approximate PCA on raw data across multiple GPUs with Dask-CUDA."""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available but raw GPU mode was requested.")
    if not DASK_CUDA_AVAILABLE:
        raise ImportError(
            "dask-cuda is not available but multi-GPU raw mode was requested."
        )
    if raw_downsample_size < 1:
        raise ValueError(f"raw_downsample_size must be >= 1, got {raw_downsample_size}")
    if raw_projection_dim < n_components:
        raise ValueError(
            f"raw_projection_dim ({raw_projection_dim}) must be >= n_components ({n_components})."
        )
    if wait is None:
        raise ImportError("dask.distributed.wait is unavailable.")

    logger.info(
        "Starting Dask CUDA cluster with %d GPUs for raw approximate PCA...", num_gpus
    )
    print(num_gpus)
    cluster = LocalCUDACluster(n_workers=num_gpus, threads_per_worker=4)
    client = Client(cluster)
    logger.info("Dask dashboard available at: %s", client.dashboard_link)

    try:
        active_workers = _wait_for_requested_workers(client, num_gpus)
        logger.info("Using %d Dask CUDA workers for raw PCA.", len(active_workers))

        rng = np.random.default_rng(seed)
        feature_indices: Optional[np.ndarray] = None
        max_inflight = max(2 * len(active_workers), len(active_workers))
        submitted_samples = 0

        # Fit pass: initialize worker-local accumulators once, then stream shard tasks.
        inflight = []
        with tqdm(
            loader, desc="Fitting PCA (raw approx, multi-GPU)", unit="batch"
        ) as pbar:
            for batch in pbar:
                raw_batch = _batch_to_numpy(batch)
                if raw_batch.ndim != 5:
                    raise ValueError(
                        f"Expected raw batch with 5 dims (B,C,D,H,W), got shape {tuple(raw_batch.shape)}"
                    )

                if feature_indices is None:
                    projected_feature_count = int(raw_batch.shape[1]) * (
                        raw_downsample_size**3
                    )
                    if raw_projection_dim > projected_feature_count:
                        raise ValueError(
                            f"raw_projection_dim ({raw_projection_dim}) must be <= projected feature count ({projected_feature_count})."
                        )
                    feature_indices = np.sort(
                        rng.choice(
                            projected_feature_count,
                            size=raw_projection_dim,
                            replace=False,
                        )
                    ).astype(np.int64, copy=False)

                    init_futures = [
                        client.submit(
                            _raw_gpu_worker_reset_state,
                            feature_indices,
                            raw_downsample_size,
                            workers=[worker],
                            allow_other_workers=False,
                            pure=False,
                        )
                        for worker in active_workers
                    ]
                    client.gather(init_futures)

                shards = [
                    np.asarray(shard, dtype=np.float32, order="C")
                    for shard in np.array_split(raw_batch, len(active_workers), axis=0)
                    if shard.shape[0] > 0
                ]
                for shard_idx, shard in enumerate(shards):
                    worker = active_workers[shard_idx % len(active_workers)]
                    inflight.append(
                        client.submit(
                            _raw_gpu_worker_accumulate_shard,
                            shard,
                            feature_indices,
                            raw_downsample_size,
                            workers=[worker],
                            allow_other_workers=False,
                            pure=False,
                        )
                    )

                while len(inflight) >= max_inflight:
                    done, not_done = wait(inflight, return_when="FIRST_COMPLETED")
                    done_list = list(done)
                    inflight = list(not_done)
                    submitted_samples += sum(int(v) for v in client.gather(done_list))
                    pbar.set_postfix(
                        {"samples": submitted_samples, "inflight": len(inflight)}
                    )

            while inflight:
                done, not_done = wait(inflight, return_when="FIRST_COMPLETED")
                done_list = list(done)
                inflight = list(not_done)
                submitted_samples += sum(int(v) for v in client.gather(done_list))
                pbar.set_postfix(
                    {"samples": submitted_samples, "inflight": len(inflight)}
                )

        if feature_indices is None:
            raise ValueError("No raw data batches were available for PCA.")

        state_futures = [
            client.submit(
                _raw_gpu_worker_get_state,
                raw_projection_dim,
                workers=[worker],
                allow_other_workers=False,
                pure=False,
            )
            for worker in active_workers
        ]
        total_samples = 0
        sum_total = np.zeros((raw_projection_dim,), dtype=np.float64)
        gram_total = np.zeros(
            (raw_projection_dim, raw_projection_dim), dtype=np.float64
        )
        for n_local, sum_local, gram_local in client.gather(state_futures):
            total_samples += int(n_local)
            sum_total += sum_local
            gram_total += gram_local

        if total_samples != submitted_samples:
            logger.warning(
                "Sample count mismatch between submitted tasks (%d) and worker accumulators (%d); using worker accumulator count.",
                submitted_samples,
                total_samples,
            )

        mean_proj, components_proj, explained_variance_ratio, singular_values = (
            _compute_pca_from_projected_covariance(
                gram_total=gram_total,
                sum_total=sum_total,
                total_samples=total_samples,
                n_components=n_components,
                device_id=0,
            )
        )

        # Transform pass: pipeline submissions and gather asynchronously while preserving sample order.
        logger.info("Transforming raw data across GPUs to collect component scores.")
        components_list = []
        transformed_samples = 0
        inflight = []
        pending_meta = {}
        expected_shards = {}
        ready_scores = {}
        next_batch_to_emit = 0

        def drain_completed(done_futures: list) -> None:
            nonlocal transformed_samples, next_batch_to_emit
            done_results = client.gather(done_futures)
            for fut, scores in zip(done_futures, done_results):
                batch_idx, shard_idx = pending_meta.pop(fut)
                if batch_idx not in ready_scores:
                    ready_scores[batch_idx] = {}
                ready_scores[batch_idx][shard_idx] = scores

            while next_batch_to_emit in expected_shards:
                shard_count = expected_shards[next_batch_to_emit]
                if shard_count == 0:
                    next_batch_to_emit += 1
                    continue
                if next_batch_to_emit not in ready_scores:
                    break
                shard_map = ready_scores[next_batch_to_emit]
                if len(shard_map) < shard_count:
                    break
                for shard_idx in range(shard_count):
                    scores = shard_map[shard_idx]
                    components_list.append(scores)
                    transformed_samples += int(scores.shape[0])
                del ready_scores[next_batch_to_emit]
                next_batch_to_emit += 1

        with tqdm(
            loader, desc="Transforming (raw approx, multi-GPU)", unit="batch"
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                raw_batch = _batch_to_numpy(batch)
                shards = [
                    np.asarray(shard, dtype=np.float32, order="C")
                    for shard in np.array_split(raw_batch, len(active_workers), axis=0)
                    if shard.shape[0] > 0
                ]
                expected_shards[batch_idx] = len(shards)

                for shard_idx, shard in enumerate(shards):
                    worker = active_workers[shard_idx % len(active_workers)]
                    fut = client.submit(
                        _raw_gpu_worker_transform_shard,
                        shard,
                        feature_indices,
                        raw_downsample_size,
                        mean_proj,
                        components_proj,
                        workers=[worker],
                        allow_other_workers=False,
                        pure=False,
                    )
                    inflight.append(fut)
                    pending_meta[fut] = (batch_idx, shard_idx)

                while len(inflight) >= max_inflight:
                    done, not_done = wait(inflight, return_when="FIRST_COMPLETED")
                    done_list = list(done)
                    inflight = list(not_done)
                    drain_completed(done_list)
                    pbar.set_postfix(
                        {"samples": transformed_samples, "inflight": len(inflight)}
                    )

            while inflight:
                done, not_done = wait(inflight, return_when="FIRST_COMPLETED")
                done_list = list(done)
                inflight = list(not_done)
                drain_completed(done_list)
                pbar.set_postfix(
                    {"samples": transformed_samples, "inflight": len(inflight)}
                )

            # Emit any trailing empty batches.
            while (
                next_batch_to_emit in expected_shards
                and expected_shards[next_batch_to_emit] == 0
            ):
                next_batch_to_emit += 1

        if pending_meta:
            raise RuntimeError("Transform futures were not fully consumed.")
        if next_batch_to_emit != len(expected_shards):
            raise RuntimeError(
                f"Transform ordering mismatch: emitted {next_batch_to_emit} / {len(expected_shards)} batches."
            )

        components = np.concatenate(components_list, axis=0)
        result = _build_pca_result(
            explained_variance_ratio=explained_variance_ratio,
            singular_values=singular_values,
            n_components=n_components,
            total_samples=total_samples,
        )
        return result, components
    finally:
        client.close()
        cluster.close()
        logger.info("Dask cluster shut down.")


def _run_pca_single_gpu(
    loader: DataLoader,
    n_components: int,
    batch_size: int,
    gpu_id: int = 0,
) -> tuple:
    """Run IncrementalPCA on a single GPU using cuML."""
    if not CUML_AVAILABLE:
        raise ImportError(
            "cuML is not available. Install RAPIDS with: "
            "pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12"
        )

    with cp.cuda.Device(gpu_id):
        ipca = cuIPCA(n_components=n_components, batch_size=batch_size)

        # First pass: partial_fit
        total_samples = 0
        with tqdm(loader, desc=f"Fitting PCA (GPU:{gpu_id})", unit="batch") as pbar:
            for batch in pbar:
                flat = _flatten_batch_cupy(batch, device_id=gpu_id)
                ipca.partial_fit(flat)
                total_samples += flat.shape[0]
                pbar.set_postfix({"samples": total_samples})
        logger.info(
            "cuML IncrementalPCA partial_fit complete over %d samples.", total_samples
        )

        # Second pass: transform
        logger.info(
            "Running cuML IncrementalPCA.transform to collect component scores."
        )
        components_list = []
        with tqdm(loader, desc=f"Transforming (GPU:{gpu_id})", unit="batch") as pbar:
            for batch in pbar:
                flat = _flatten_batch_cupy(batch, device_id=gpu_id)
                transformed = ipca.transform(flat)
                # Move back to CPU/numpy for plotting
                components_list.append(cp.asnumpy(transformed))
                pbar.set_postfix({"samples": sum(c.shape[0] for c in components_list)})

        components = np.concatenate(components_list, axis=0)

        # Extract sklearn-compatible attributes
        explained_variance_ratio = cp.asnumpy(ipca.explained_variance_ratio_)
        singular_values = cp.asnumpy(ipca.singular_values_)

    # Create a simple namespace to mimic sklearn ipca for variance info
    class PCAResult:
        pass

    result = PCAResult()
    result.explained_variance_ratio_ = explained_variance_ratio
    result.singular_values_ = singular_values
    result.n_components_ = n_components
    result.n_samples_seen_ = total_samples

    return result, components


def _run_pca_multi_gpu(
    loader: DataLoader,
    n_components: int,
    batch_size: int,
    num_gpus: int,
) -> tuple:
    """
    Run IncrementalPCA across multiple GPUs using Dask-CUDA.

    Strategy: Distribute batches across GPUs in a round-robin fashion,
    with each GPU running its own IncrementalPCA. Then merge results.

    Note: True distributed IncrementalPCA is complex; we use a simpler
    approach of parallel data loading + single-GPU PCA on aggregated data.
    For very large datasets, we accumulate data in chunks across GPUs.
    """
    if not CUML_AVAILABLE:
        raise ImportError(
            "cuML is not available. Install RAPIDS with: "
            "pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12"
        )
    if not DASK_CUDA_AVAILABLE:
        raise ImportError(
            "dask-cuda is not available. Install with: pip install dask-cuda"
        )

    logger.info("Starting Dask CUDA cluster with %d GPUs...", num_gpus)
    cluster = LocalCUDACluster(n_workers=num_gpus, threads_per_worker=1)
    client = Client(cluster)
    logger.info("Dask dashboard available at: %s", client.dashboard_link)

    try:
        # Stream one batch at a time to GPU 0 to avoid OOM (e.g. raw 3D volumes
        # can be ~1 GB per batch; accumulating all batches would exceed GPU memory).
        logger.info(
            "Streaming batches to GPU 0 for IncrementalPCA (one batch at a time to avoid OOM)."
        )
        with cp.cuda.Device(0):
            ipca = cuIPCA(n_components=n_components, batch_size=batch_size)
            total_samples = 0

            with tqdm(loader, desc="Fitting PCA (streaming)", unit="batch") as pbar:
                for batch in pbar:
                    flat = _flatten_batch_cupy(batch, device_id=0)
                    ipca.partial_fit(flat)
                    total_samples += flat.shape[0]
                    pbar.set_postfix({"samples": total_samples})
                    del flat  # free GPU memory before next batch

            logger.info(
                "cuML IncrementalPCA partial_fit complete over %d samples.",
                total_samples,
            )

            # Transform pass: stream again so we never hold all data on GPU
            logger.info("Transforming data to extract components (streaming)...")
            components_list = []
            with tqdm(loader, desc="Transforming", unit="batch") as pbar:
                for batch in pbar:
                    flat = _flatten_batch_cupy(batch, device_id=0)
                    transformed = ipca.transform(flat)
                    components_list.append(cp.asnumpy(transformed))
                    del flat
                    pbar.set_postfix(
                        {"samples": sum(c.shape[0] for c in components_list)}
                    )

            components = np.concatenate(components_list, axis=0)
            explained_variance_ratio = cp.asnumpy(ipca.explained_variance_ratio_)
            singular_values = cp.asnumpy(ipca.singular_values_)

    finally:
        client.close()
        cluster.close()
        logger.info("Dask cluster shut down.")

    class PCAResult:
        pass

    result = PCAResult()
    result.explained_variance_ratio_ = explained_variance_ratio
    result.singular_values_ = singular_values
    result.n_components_ = n_components
    result.n_samples_seen_ = total_samples

    return result, components


def run_latent_pca(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    # Use higher prefetch factor for GPU to keep it fed
    prefetch_factor = 4 if args.use_gpu else 2

    mode = _detect_store_mode(args.latents_zarr)
    if args.use_filter_cache and mode != "raw":
        raise ValueError("--use_filter_cache is supported only for raw input stores.")
    if args.use_filter_cache and not args.filter_cache_path:
        raise ValueError(
            "--filter_cache_path is required when --use_filter_cache is enabled."
        )

    filter_selection_info: Optional[Dict[str, Any]] = None
    selected_indices: Optional[list[int]] = None
    if args.use_filter_cache:
        filter_cache_path = Path(args.filter_cache_path)
        filter_stats = load_filter_stats_cache(
            cache_path=filter_cache_path,
            expected_train_size=None,
            expected_format=3,
            strict=True,
        )
        if filter_stats is None:
            raise ValueError(f"Unable to load filter cache: {filter_cache_path}")
        filter_selection_info = select_indices_from_filter_stats(
            filter_stats=filter_stats,
            susc_active_threshold_log10=float(args.susc_active_threshold_log10),
            min_active_frac=float(args.min_active_frac),
            low_info_keep_prob=float(args.low_info_keep_prob),
            seed=int(args.seed),
        )
        selected_indices = filter_selection_info["selected_indices"]
        logger.info(
            "Using filter cache: selected %d / %d cached samples.",
            len(selected_indices),
            int(filter_selection_info["cache_sample_count"]),
        )
    elif mode == "raw":
        logger.info("Filter cache disabled: using unfiltered raw sample indices.")

    logger.info("Building dataloader over Zarr: %s", args.latents_zarr)
    loader, mode = _build_dataloader(
        zarr_path=args.latents_zarr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        sample_stride=args.sample_stride,
        raw_field=args.raw_field,
        selected_indices=selected_indices,
        prefetch_factor=prefetch_factor,
    )
    logger.info("Detected input type: %s", mode)
    if mode == "raw":
        logger.info("Raw mode field selection: %s", args.raw_field)
    elif args.raw_field != "both":
        logger.info(
            "Ignoring --raw_field=%s because input type is latent.",
            args.raw_field,
        )

    # Run PCA based on mode
    if mode == "raw":
        if args.use_gpu:
            if not CUPY_AVAILABLE:
                raise ImportError(
                    "CuPy is not available but --use_gpu was specified for raw mode."
                )
            if args.num_gpus > 1 and not DASK_CUDA_AVAILABLE:
                raise ImportError(
                    "dask-cuda is not available but --num_gpus > 1 was specified for raw mode."
                )
            logger.info(
                "Raw GPU mode enabled: using %d GPU(s), downsample=%d^3, projection_dim=%d, n_components=%d.",
                args.num_gpus if args.num_gpus > 1 else 1,
                args.raw_downsample_size,
                args.raw_projection_dim,
                args.n_components,
            )
            if args.num_gpus > 1:
                ipca, components = _run_pca_raw_approx_multi_gpu(
                    loader=loader,
                    n_components=args.n_components,
                    raw_downsample_size=args.raw_downsample_size,
                    raw_projection_dim=args.raw_projection_dim,
                    seed=args.seed,
                    num_gpus=args.num_gpus,
                )
                run_num_gpus = args.num_gpus
            else:
                ipca, components = _run_pca_raw_approx_single_gpu(
                    loader=loader,
                    n_components=args.n_components,
                    raw_downsample_size=args.raw_downsample_size,
                    raw_projection_dim=args.raw_projection_dim,
                    seed=args.seed,
                    gpu_id=args.gpu_id,
                )
                run_num_gpus = 1
            run_mode = "gpu"
        else:
            logger.info(
                "Running raw approximate PCA over %d batches on CPU: downsample=%d^3, projection_dim=%d, n_components=%d.",
                len(loader),
                args.raw_downsample_size,
                args.raw_projection_dim,
                args.n_components,
            )
            ipca, components = _run_pca_raw_approx_cpu(
                loader=loader,
                n_components=args.n_components,
                batch_size=args.batch_size,
                raw_downsample_size=args.raw_downsample_size,
                raw_projection_dim=args.raw_projection_dim,
                seed=args.seed,
            )
            run_mode = "cpu"
            run_num_gpus = 0
    else:
        if args.use_gpu:
            if not CUML_AVAILABLE:
                raise ImportError(
                    "cuML is not available but --use_gpu was specified. "
                    "Install RAPIDS with: pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12"
                )
            if args.num_gpus > 1 and not DASK_CUDA_AVAILABLE:
                raise ImportError(
                    "dask-cuda is not available but --num_gpus > 1 was specified. "
                    "Install with: pip install dask-cuda"
                )
            logger.info(
                "GPU mode enabled: using %d GPU(s) with cuML",
                args.num_gpus if args.num_gpus > 1 else 1,
            )
            logger.info(
                "Fitting IncrementalPCA with n_components=%d over %d batches (batch_size=%d).",
                args.n_components,
                len(loader),
                args.batch_size,
            )
            if args.num_gpus > 1:
                ipca, components = _run_pca_multi_gpu(
                    loader, args.n_components, args.batch_size, args.num_gpus
                )
            else:
                ipca, components = _run_pca_single_gpu(
                    loader, args.n_components, args.batch_size, args.gpu_id
                )
            run_mode = "gpu"
            run_num_gpus = args.num_gpus
        else:
            logger.info("CPU mode: using sklearn IncrementalPCA")
            logger.info(
                "Fitting IncrementalPCA with n_components=%d over %d batches (batch_size=%d).",
                args.n_components,
                len(loader),
                args.batch_size,
            )
            ipca, components = _run_pca_cpu(loader, args.n_components, args.batch_size)
            run_mode = "cpu"
            run_num_gpus = 0

    if components.size == 0:
        logger.error("No data was collected for PCA; check dataset and filters.")
        return

    logger.info("Collected component scores with shape %s.", components.shape)

    # Save explained variance info
    variance_info = {
        "explained_variance_ratio": ipca.explained_variance_ratio_.tolist(),
        "singular_values": ipca.singular_values_.tolist(),
        "n_components": int(ipca.n_components_),
        "n_samples_seen": int(ipca.n_samples_seen_),
        "mode": run_mode,
        "num_gpus": run_num_gpus,
        "raw_field": args.raw_field if mode == "raw" else None,
        "filter_enabled": bool(args.use_filter_cache and mode == "raw"),
        "filter_cache_path": args.filter_cache_path if args.use_filter_cache else None,
        "susc_active_threshold_log10": (
            float(args.susc_active_threshold_log10)
            if args.use_filter_cache and mode == "raw"
            else None
        ),
        "min_active_frac": (
            float(args.min_active_frac) if args.use_filter_cache and mode == "raw" else None
        ),
        "low_info_keep_prob": (
            float(args.low_info_keep_prob)
            if args.use_filter_cache and mode == "raw"
            else None
        ),
        "filtered_original_count": (
            int(filter_selection_info["cache_sample_count"])
            if filter_selection_info is not None
            else None
        ),
        "filtered_selected_count": (
            int(len(filter_selection_info["selected_indices"]))
            if filter_selection_info is not None
            else None
        ),
    }
    with (output_dir / "pca_variance.json").open("w") as f:
        json.dump(variance_info, f, indent=2)
    logger.info("Saved PCA variance info to %s", output_dir / "pca_variance.json")

    # Scree plot
    _plot_scree(ipca.explained_variance_ratio_, output_dir / "pca_scree.png")

    # 2D scatter plots for a few component pairs
    scatter_pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in scatter_pairs:
        scatter_path = output_dir / f"pca_scatter_pc{i+1}_pc{j+1}.png"
        _plot_scatter(
            components,
            i=i,
            j=j,
            output_path=scatter_path,
            max_points=args.scatter_max_points,
            seed=args.seed,
        )

    # Histograms for first few PCs
    num_hist_pcs = min(5, components.shape[1])
    for pc_index in range(num_hist_pcs):
        hist_path = output_dir / f"pca_hist_pc{pc_index+1}.png"
        _plot_pc_histogram(components, pc_index=pc_index, output_path=hist_path)

    logger.info("Latent PCA analysis complete. Results written to %s", output_dir)


def main() -> None:
    args = parse_args()
    run_latent_pca(args)


if __name__ == "__main__":
    main()
