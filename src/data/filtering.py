"""Shared helpers for cache-backed sample filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _required_filter_cache_keys() -> set[str]:
    return {
        "cache_format_version",
        "sample_indices",
        "rock_id_counts",
        "susc_log10_lut",
        "voxel_count_per_sample",
    }


def load_filter_stats_cache(
    cache_path: Path,
    expected_train_size: Optional[int] = None,
    expected_format: int = 3,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load and validate a train-filter cache payload.

    Returns:
        Valid payload dict if cache is present and valid, else None (unless strict=True).
    """
    if not cache_path.exists():
        if strict:
            raise FileNotFoundError(f"Filter cache not found: {cache_path}")
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        if strict:
            raise ValueError(f"Invalid filter cache payload type: {type(payload)!r}")
        return None

    missing_keys = sorted(k for k in _required_filter_cache_keys() if k not in payload)
    if missing_keys:
        if strict:
            raise ValueError(
                f"Filter cache missing required keys ({', '.join(missing_keys)}): {cache_path}"
            )
        return None

    cached_format = int(payload.get("cache_format_version", -1))
    if cached_format != int(expected_format):
        if strict:
            raise ValueError(
                "Filter cache format version mismatch "
                f"({cached_format} vs {int(expected_format)}): {cache_path}"
            )
        return None

    if expected_train_size is not None:
        cached_train_size = int(payload.get("train_size", -1))
        if cached_train_size != int(expected_train_size):
            if strict:
                raise ValueError(
                    "Filter cache train_size mismatch "
                    f"({cached_train_size} vs {int(expected_train_size)}): {cache_path}"
                )
            return None

    return payload


def select_indices_from_filter_stats(
    filter_stats: Dict[str, Any],
    susc_active_threshold_log10: float,
    min_active_frac: float,
    low_info_keep_prob: float,
    seed: int,
) -> Dict[str, Any]:
    """Apply train filter policy to cache stats and return selected sample indices."""
    if min_active_frac < 0.0 or min_active_frac > 1.0:
        raise ValueError("min_active_frac must be in [0, 1]")
    if low_info_keep_prob < 0.0 or low_info_keep_prob > 1.0:
        raise ValueError("low_info_keep_prob must be in [0, 1]")

    sample_indices = torch.as_tensor(filter_stats["sample_indices"], dtype=torch.int64)
    rock_id_counts = torch.as_tensor(filter_stats["rock_id_counts"], dtype=torch.int64)
    susc_log10_lut = torch.as_tensor(filter_stats["susc_log10_lut"], dtype=torch.float32)
    voxel_count_per_sample = int(filter_stats["voxel_count_per_sample"])
    if voxel_count_per_sample <= 0:
        raise ValueError(
            f"voxel_count_per_sample must be > 0, got {voxel_count_per_sample}"
        )

    threshold = float(susc_active_threshold_log10)
    active_mask = susc_log10_lut > threshold
    active_counts = (
        rock_id_counts.to(dtype=torch.float32) * active_mask.to(dtype=torch.float32)
    ).sum(dim=1)

    active_frac = active_counts.to(dtype=torch.float32) / float(voxel_count_per_sample)
    informative_mask = active_frac >= float(min_active_frac)
    informative_indices = sample_indices[informative_mask]
    low_info_indices = sample_indices[~informative_mask]

    informative_count = int(informative_mask.sum().item())
    low_info_count = int((~informative_mask).sum().item())

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    retained_low_info_count = 0
    retained_low_info_indices = torch.empty(0, dtype=torch.int64)
    if low_info_count > 0 and low_info_keep_prob > 0.0:
        keep_mask = torch.rand(low_info_count, generator=rng) < float(low_info_keep_prob)
        retained_low_info_indices = low_info_indices[keep_mask]
        retained_low_info_count = int(keep_mask.sum().item())

    selected_indices_tensor = torch.cat(
        [informative_indices, retained_low_info_indices], dim=0
    )
    selected_indices = selected_indices_tensor.tolist()

    original_count = int(filter_stats.get("train_size", int(sample_indices.numel())))
    return {
        "selected_indices": selected_indices,
        "original_count": original_count,
        "informative_count": informative_count,
        "low_info_count": low_info_count,
        "retained_low_info_count": retained_low_info_count,
        "cache_sample_count": int(sample_indices.numel()),
    }
