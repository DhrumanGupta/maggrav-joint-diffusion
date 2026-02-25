"""Density-aware latent weighting filters."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

# Use faiss-gpu for installing:
# conda install -c pytorch faiss-gpu
# faiss-cpu would take forever to run.
import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .dataset_filters import DatasetFilter, DatasetFilterContext, DatasetSamplingPlan

Mode = Literal["auto", "cache_only", "compute_if_missing", "recompute"]
DistanceType = Literal["cosine", "l2"]
ProjectionBackend = Literal["auto", "torch_countsketch", "sklearn_sparse"]

logger = logging.getLogger(__name__)


class LatentDensityKNNWeighter(DatasetFilter[Any]):
    """Compute density-aware sample weights for latent datasets.

    This filter keeps all samples and returns per-sample weights derived from local
    spacing in projected latent space.
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        cache_path: Optional[str] = None,
        mode: Mode = "auto",
        proj_dim: int = 4096,
        proj_seed: int = 0,
        k: int = 30,
        alpha: float = 1.5,
        w_min: float = 0.05,
        w_max: float = 5.0,
        distance: DistanceType = "cosine",
        normalize_embeddings: bool = True,
        index_factory: str = "IVF16384,PQ64x8",
        nprobe: int = 64,
        train_size: int = 200000,
        batch_size: Optional[int] = None,
        search_extra: int = 8,
        use_gpu: bool = True,
        faiss_num_gpus: int = 0,
        projection_backend: ProjectionBackend = "auto",
        projection_device: Optional[str] = None,
        projection_feature_chunk: int = 16384,
        verbose: bool = True,
    ) -> None:
        if mode not in {"auto", "cache_only", "compute_if_missing", "recompute"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if distance not in {"cosine", "l2"}:
            raise ValueError(f"distance must be 'cosine' or 'l2', got {distance!r}")
        if proj_dim < 2:
            raise ValueError(f"proj_dim must be >= 2, got {proj_dim}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if w_min <= 0 or w_max <= 0:
            raise ValueError("w_min and w_max must be > 0")
        if w_min > w_max:
            raise ValueError("w_min cannot be greater than w_max")
        if nprobe < 1:
            raise ValueError(f"nprobe must be >= 1, got {nprobe}")
        if train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {train_size}")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if search_extra < 0:
            raise ValueError("search_extra must be >= 0")
        if faiss_num_gpus < 0:
            raise ValueError("faiss_num_gpus must be >= 0")
        if projection_backend not in {"auto", "torch_countsketch", "sklearn_sparse"}:
            raise ValueError(
                f"Unsupported projection_backend: {projection_backend!r}"
            )
        if projection_feature_chunk < 1:
            raise ValueError("projection_feature_chunk must be >= 1")

        self.cache_path = cache_path
        self.mode = mode
        self.proj_dim = int(proj_dim)
        self.proj_seed = int(proj_seed)
        self.k = int(k)
        self.alpha = float(alpha)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.distance = distance
        self.normalize_embeddings = bool(normalize_embeddings)
        self.index_factory = index_factory
        self.nprobe = int(nprobe)
        self.train_size = int(train_size)
        self.batch_size = batch_size
        self.search_extra = int(search_extra)
        self.use_gpu = bool(use_gpu)
        self.faiss_num_gpus = int(faiss_num_gpus)
        self.projection_backend = projection_backend
        self.projection_device = projection_device
        self.projection_feature_chunk = int(projection_feature_chunk)
        self.verbose = bool(verbose)
        self._last_faiss_backend = "cpu"

    def _log(self, msg: str, *args: object) -> None:
        if self.verbose:
            logger.info("[LatentDensityKNNWeighter] " + msg, *args)

    def build_plan(self, context: DatasetFilterContext[Any]) -> DatasetSamplingPlan:
        source_length = int(context.source_length)
        selected = torch.arange(source_length, dtype=torch.long)
        if source_length == 0:
            return DatasetSamplingPlan(
                selected_indices=selected,
                weights=torch.empty(0, dtype=torch.float32),
            )

        resolved_mode = self._resolve_mode()
        cache_path = self._resolve_cache_path(context)
        self._log(
            "start | N=%d mode=%s distributed=%s cache=%s",
            source_length,
            resolved_mode,
            self._is_distributed(),
            cache_path,
        )
        self._log(
            "projection | backend=%s device=%s chunk=%d proj_dim=%d",
            self._resolve_projection_backend(),
            self._resolve_projection_device(),
            self.projection_feature_chunk,
            self.proj_dim,
        )
        self._log(
            "faiss | installed=%s requested_gpu=%s visible_gpus=%d target_gpus=%s index_factory=%s nprobe=%d",
            True,
            self.use_gpu,
            int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0,
            self.faiss_num_gpus if self.faiss_num_gpus > 0 else "all",
            self.index_factory,
            self.nprobe,
        )

        cached_weights = self._try_load_cached_weights(
            cache_path=cache_path,
            expected_source_length=source_length,
            expected_config=self._cache_config(),
        )
        if cached_weights is not None and resolved_mode != "recompute":
            self._log("cache hit | loaded weights from %s", cache_path)
            return DatasetSamplingPlan(
                selected_indices=selected,
                weights=cached_weights,
            )

        if resolved_mode == "cache_only":
            raise FileNotFoundError(
                "Density weight cache missing or invalid in distributed/cache-only mode: "
                f"{cache_path}. Run a single-process precompute first (mode='recompute' or "
                "mode='compute_if_missing') and reuse the generated cache for distributed training."
            )

        if self._is_distributed():
            raise RuntimeError(
                "Computing density weights is disabled in distributed mode. "
                "Use cache_only and precompute weights in a single-process run."
            )

        spacing, weights = self._compute_spacing_and_weights(context, cache_path)
        self._log(
            "done | spacing[min=%.6f mean=%.6f max=%.6f] weights[min=%.6f mean=%.6f max=%.6f]",
            float(np.min(spacing)) if spacing.size else 0.0,
            float(np.mean(spacing)) if spacing.size else 0.0,
            float(np.max(spacing)) if spacing.size else 0.0,
            float(np.min(weights)) if weights.size else 0.0,
            float(np.mean(weights)) if weights.size else 0.0,
            float(np.max(weights)) if weights.size else 0.0,
        )
        self._save_cache(
            cache_path=cache_path,
            context=context,
            spacing=spacing,
            weights=weights,
            config=self._cache_config(),
        )
        self._log("cache write | saved weights to %s", cache_path)
        return DatasetSamplingPlan(
            selected_indices=selected,
            weights=torch.from_numpy(weights.astype(np.float32, copy=False)),
        )

    def _resolve_mode(self) -> Mode:
        if self.mode != "auto":
            return cast(Mode, self.mode)
        if self._is_distributed():
            return "cache_only"
        return "compute_if_missing"

    def _is_distributed(self) -> bool:
        return (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

    def _resolve_cache_path(self, context: DatasetFilterContext[Any]) -> Path:
        if self.cache_path:
            return Path(self.cache_path).expanduser().resolve()

        zarr_path = context.metadata.get("zarr_path")
        if zarr_path is None:
            raise ValueError(
                "Dataset metadata is missing 'zarr_path'; explicit cache_path is required."
            )

        zarr_fs_path = Path(str(zarr_path)).expanduser().resolve()
        name = zarr_fs_path.name
        if name.endswith(".zarr"):
            stem = name[: -len(".zarr")]
        else:
            stem = zarr_fs_path.stem if zarr_fs_path.suffix else name
        return zarr_fs_path.parent / f"{stem}.density_weights.pt"

    def _cache_config(self) -> Dict[str, Any]:
        return {
            "proj_dim": self.proj_dim,
            "proj_seed": self.proj_seed,
            "k": self.k,
            "alpha": self.alpha,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "distance": self.distance,
            "normalize_embeddings": self.normalize_embeddings,
            "index_factory": self.index_factory,
            "nprobe": self.nprobe,
            "train_size": self.train_size,
            "search_extra": self.search_extra,
            "use_gpu": self.use_gpu,
            "faiss_num_gpus": self.faiss_num_gpus,
            "projection_backend": self._resolve_projection_backend(),
            "projection_device": self._resolve_projection_device(),
            "projection_feature_chunk": self.projection_feature_chunk,
        }

    def _try_load_cached_weights(
        self,
        cache_path: Path,
        expected_source_length: int,
        expected_config: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        if not cache_path.exists():
            return None

        payload = cast(Dict[str, Any], torch.load(cache_path, map_location="cpu"))
        if int(payload.get("version", -1)) != self.CACHE_VERSION:
            return None
        if int(payload.get("source_length", -1)) != expected_source_length:
            return None

        stored_config = payload.get("config")
        if not isinstance(stored_config, dict) or stored_config != expected_config:
            return None

        weights = torch.as_tensor(payload.get("weights"), dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() != expected_source_length:
            return None
        if not torch.isfinite(weights).all():
            return None
        if (weights < 0).any():
            return None

        return weights

    def _save_cache(
        self,
        cache_path: Path,
        context: DatasetFilterContext[Any],
        spacing: np.ndarray,
        weights: np.ndarray,
        config: Dict[str, Any],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.CACHE_VERSION,
            "created_at_unix": int(time.time()),
            "source_length": int(context.source_length),
            "sample_shape": context.sample_shape,
            "dataset_name": context.dataset_name,
            "zarr_path": str(context.metadata.get("zarr_path", "")),
            "config": config,
            "spacing": torch.from_numpy(spacing.astype(np.float32, copy=False)),
            "weights": torch.from_numpy(weights.astype(np.float32, copy=False)),
        }

        with tempfile.NamedTemporaryFile(
            dir=str(cache_path.parent),
            prefix=cache_path.name,
            suffix=".tmp",
            delete=False,
        ) as tmpf:
            tmp_path = Path(tmpf.name)

        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _compute_spacing_and_weights(
        self,
        context: DatasetFilterContext[Any],
        cache_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        source_length = int(context.source_length)
        sample_shape = context.sample_shape
        if sample_shape is None:
            raise ValueError(
                "sample_shape is required for latent density weighting projection."
            )

        backend = self._resolve_projection_backend()
        if backend == "torch_countsketch":
            self._log("stage projection | starting torch_countsketch")
            projected = self._project_with_torch_countsketch(
                context=context,
                cache_path=cache_path,
                sample_shape=sample_shape,
                source_length=source_length,
            )
        else:
            self._log("stage projection | starting sklearn_sparse")
            projected = self._project_with_sklearn_sparse(
                context=context,
                cache_path=cache_path,
                sample_shape=sample_shape,
                source_length=source_length,
            )
        self._log("stage projection | completed in %.1fs", time.time() - t0)

        t1 = time.time()
        self._log("stage faiss | starting ANN build+search")
        spacing = self._compute_spacing_with_faiss(projected)
        self._log(
            "stage faiss | completed in %.1fs backend=%s", time.time() - t1, self._last_faiss_backend
        )
        weights = self._spacing_to_weights(spacing)

        proj_path = Path(cast(np.memmap, projected).filename)
        del projected
        proj_path.unlink(missing_ok=True)
        return spacing, weights

    def _resolve_projection_backend(self) -> Literal["torch_countsketch", "sklearn_sparse"]:
        if self.projection_backend == "auto":
            return "torch_countsketch" if torch.cuda.is_available() else "sklearn_sparse"
        return cast(Literal["torch_countsketch", "sklearn_sparse"], self.projection_backend)

    def _resolve_projection_device(self) -> str:
        if self.projection_device is not None:
            return self.projection_device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _project_with_sklearn_sparse(
        self,
        context: DatasetFilterContext[Any],
        cache_path: Path,
        sample_shape: tuple[int, ...],
        source_length: int,
    ) -> np.memmap:
        try:
            from sklearn.random_projection import SparseRandomProjection
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required to compute density weights."
            ) from exc

        input_dim = int(np.prod(sample_shape))

        projector = SparseRandomProjection(
            n_components=self.proj_dim,
            random_state=self.proj_seed,
            dense_output=True,
        )
        projector.fit(np.zeros((1, input_dim), dtype=np.float32))

        proj_path = cache_path.with_suffix(cache_path.suffix + ".projected.tmp")
        projected = np.memmap(
            proj_path,
            mode="w+",
            dtype=np.float32,
            shape=(source_length, self.proj_dim),
        )

        batch_size = (
            context.default_batch_size if self.batch_size is None else self.batch_size
        )
        cursor = 0
        for index_batch in context.iter_index_batches(batch_size=batch_size):
            batch = context.get_source_samples(index_batch)
            mu = self._extract_mu(batch)
            flat = mu.reshape(mu.shape[0], -1).detach().cpu().numpy()
            y = projector.transform(flat)
            y = np.asarray(y, dtype=np.float32)
            if self.normalize_embeddings:
                norms = np.linalg.norm(y, axis=1, keepdims=True)
                y = y / np.maximum(norms, 1e-12)
            b = y.shape[0]
            projected[cursor : cursor + b] = y
            cursor += b

        if cursor != source_length:
            raise RuntimeError(
                f"Projection write mismatch: expected {source_length}, wrote {cursor}."
            )

        projected.flush()
        return projected

    def _project_with_torch_countsketch(
        self,
        context: DatasetFilterContext[Any],
        cache_path: Path,
        sample_shape: tuple[int, ...],
        source_length: int,
    ) -> np.memmap:
        input_dim = int(np.prod(sample_shape))
        proj_path = cache_path.with_suffix(cache_path.suffix + ".projected.tmp")
        projected = np.memmap(
            proj_path,
            mode="w+",
            dtype=np.float32,
            shape=(source_length, self.proj_dim),
        )

        device = torch.device(self._resolve_projection_device())
        self._log("projection device resolved to %s", device)
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(self.proj_seed)

        bucket_ids = torch.randint(
            low=0,
            high=self.proj_dim,
            size=(input_dim,),
            generator=cpu_gen,
            dtype=torch.long,
        )
        sign_bits = torch.randint(
            low=0,
            high=2,
            size=(input_dim,),
            generator=cpu_gen,
            dtype=torch.int8,
        )
        signs = sign_bits.to(dtype=torch.float32).mul_(2.0).sub_(1.0)

        bucket_ids = bucket_ids.to(device=device, non_blocking=True)
        signs = signs.to(device=device, non_blocking=True)

        batch_size = (
            context.default_batch_size if self.batch_size is None else self.batch_size
        )
        cursor = 0
        feature_chunk = self.projection_feature_chunk
        for index_batch in context.iter_index_batches(batch_size=batch_size):
            batch = context.get_source_samples(index_batch)
            mu = self._extract_mu(batch)
            flat = mu.reshape(mu.shape[0], -1)
            if int(flat.shape[1]) != input_dim:
                raise ValueError(
                    f"Unexpected latent flat dim {flat.shape[1]} (expected {input_dim})."
                )

            x = flat.to(device=device, dtype=torch.float32, non_blocking=True)
            b = int(x.shape[0])
            y = torch.zeros((b, self.proj_dim), dtype=torch.float32, device=device)

            for start in range(0, input_dim, feature_chunk):
                end = min(start + feature_chunk, input_dim)
                chunk_ids = bucket_ids[start:end]
                chunk_signs = signs[start:end]
                x_chunk = x[:, start:end] * chunk_signs.unsqueeze(0)
                y.scatter_add_(1, chunk_ids.unsqueeze(0).expand(b, -1), x_chunk)

            if self.normalize_embeddings:
                y = F.normalize(y, p=2, dim=1, eps=1e-12)

            y_cpu = y.detach().cpu().numpy().astype(np.float32, copy=False)
            projected[cursor : cursor + b] = y_cpu
            cursor += b

        if cursor != source_length:
            raise RuntimeError(
                f"Projection write mismatch: expected {source_length}, wrote {cursor}."
            )

        projected.flush()
        return projected

    def _extract_mu(self, batch: Any) -> torch.Tensor:
        if not hasattr(batch, "mu"):
            raise TypeError(
                "LatentDensityKNNWeighter expects batch objects with a 'mu' tensor."
            )
        mu = getattr(batch, "mu")
        if not isinstance(mu, torch.Tensor):
            raise TypeError(f"Expected batch.mu to be a torch.Tensor, got {type(mu)!r}")
        if mu.ndim < 2:
            raise ValueError(f"Expected batch.mu with ndim>=2, got {mu.ndim}")
        return mu.to(dtype=torch.float32)

    def _compute_spacing_with_faiss(self, projected: np.ndarray) -> np.ndarray:
        n = int(projected.shape[0])
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if n == 1:
            return np.ones(1, dtype=np.float32)

        metric = (
            faiss.METRIC_INNER_PRODUCT if self.distance == "cosine" else faiss.METRIC_L2
        )
        cpu_index = faiss.index_factory(self.proj_dim, self.index_factory, metric)
        index_for_search = self._maybe_move_index_to_gpu(cpu_index)

        if not index_for_search.is_trained:
            train_n = min(self.train_size, n)
            self._log("faiss train | train_n=%d", train_n)
            rng = np.random.default_rng(self.proj_seed)
            train_ids = rng.choice(n, size=train_n, replace=False)
            train_x = np.asarray(projected[train_ids], dtype=np.float32)
            index_for_search.train(train_x)

        if hasattr(index_for_search, "nprobe"):
            index_for_search.nprobe = self.nprobe

        ann_batch = 8192
        k_eff = min(self.k, n - 1)
        search_k = min(k_eff + 1 + self.search_extra, n)
        self._log(
            "faiss add/search | batch=%d k=%d search_k=%d",
            ann_batch,
            k_eff,
            search_k,
        )
        for start in range(0, n, ann_batch):
            end = min(start + ann_batch, n)
            x = np.asarray(projected[start:end], dtype=np.float32)
            index_for_search.add(x)

        spacing = np.zeros((n,), dtype=np.float32)
        for start in range(0, n, ann_batch):
            end = min(start + ann_batch, n)
            q = np.asarray(projected[start:end], dtype=np.float32)
            dists, nbrs = index_for_search.search(q, search_k)
            qids = np.arange(start, end, dtype=np.int64)
            self._accumulate_spacing_rows(
                out_spacing=spacing,
                query_ids=qids,
                neighbor_ids=nbrs,
                neighbor_scores=dists,
                k_eff=k_eff,
            )

        return spacing

    def _maybe_move_index_to_gpu(self, cpu_index: Any) -> Any:
        if not self.use_gpu:
            self._last_faiss_backend = "cpu (disabled)"
            return cpu_index

        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
            self._last_faiss_backend = "cpu (no faiss GPUs)"
            return cpu_index

        try:
            ngpu_available = int(faiss.get_num_gpus())
            ngpu_target = ngpu_available
            if self.faiss_num_gpus > 0:
                ngpu_target = min(self.faiss_num_gpus, ngpu_available)

            if ngpu_target > 1:
                co = faiss.GpuMultipleClonerOptions()
                # Replicas usually maximize search throughput; sharding minimizes memory.
                co.shard = False
                co.useFloat16 = True

                if hasattr(faiss, "index_cpu_to_gpu_multiple_py") and hasattr(
                    faiss,
                    "StandardGpuResources",
                ):
                    resources = [
                        faiss.StandardGpuResources() for _ in range(ngpu_target)
                    ]
                    gpu_ids = list(range(ngpu_target))
                    moved = faiss.index_cpu_to_gpu_multiple_py(
                        resources,
                        cpu_index,
                        co=co,
                        gpus=gpu_ids,
                    )
                    self._last_faiss_backend = f"gpu(multi={ngpu_target},replica)"
                    return moved

                if hasattr(faiss, "index_cpu_to_all_gpus"):
                    moved = faiss.index_cpu_to_all_gpus(cpu_index)
                    self._last_faiss_backend = f"gpu(all={ngpu_target})"
                    return moved

            if hasattr(faiss, "StandardGpuResources") and hasattr(
                faiss,
                "index_cpu_to_gpu",
            ):
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                moved = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
                self._last_faiss_backend = "gpu(single=0)"
                return moved
        except Exception:
            self._last_faiss_backend = "cpu (gpu transfer failed)"
            return cpu_index

        self._last_faiss_backend = "cpu (fallback)"
        return cpu_index

    def _accumulate_spacing_rows(
        self,
        out_spacing: np.ndarray,
        query_ids: np.ndarray,
        neighbor_ids: np.ndarray,
        neighbor_scores: np.ndarray,
        k_eff: int,
    ) -> None:
        for row, qid in enumerate(query_ids):
            ids = neighbor_ids[row]
            vals = neighbor_scores[row]

            keep = ids != int(qid)
            vals = vals[keep]
            if vals.size == 0:
                out_spacing[int(qid)] = 0.0
                continue

            vals = vals[:k_eff]
            if self.distance == "cosine":
                dist_vals = 1.0 - vals
                dist_vals = np.clip(dist_vals, 0.0, 2.0)
            else:
                dist_vals = np.sqrt(np.clip(vals, 0.0, None))

            out_spacing[int(qid)] = float(np.mean(dist_vals, dtype=np.float32))

    def _spacing_to_weights(self, spacing: np.ndarray) -> np.ndarray:
        if spacing.size == 0:
            return spacing.astype(np.float32, copy=False)

        spacing = np.asarray(spacing, dtype=np.float32)
        median = float(np.median(spacing))
        median = max(median, 1e-12)
        scaled = np.power(np.maximum(spacing, 1e-12) / median, self.alpha)
        clipped = np.clip(scaled, self.w_min, self.w_max).astype(np.float32)
        mean_w = float(np.mean(clipped))
        mean_w = max(mean_w, 1e-12)
        return clipped / mean_w
