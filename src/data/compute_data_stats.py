"""
Compute distribution statistics for density + susceptibility volumes.

Usage:
    python src/compute_data_stats.py --zarr_path /path/to/zarr
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from data.zarr_dataset import NoddyverseZarrDataset

# Tell torch to use CPU only.
torch.set_default_device("cpu")


class JointDensitySuscDataset(Dataset):
    def __init__(self, zarr_path: str) -> None:
        self.base = NoddyverseZarrDataset(
            zarr_path,
            fields=("rock_types",),
            include_metadata=False,
            return_tensors=True,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.base[idx]
        density = sample["density"]
        susceptibility = sample["susceptibility"]
        return torch.stack([density, susceptibility], dim=0)


def _compute_moments(values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    values = values.to(dtype=torch.float64)
    count = torch.tensor(values.numel(), dtype=torch.float64)
    mean = values.mean()
    centered = values - mean
    m2 = torch.sum(centered**2)
    m3 = torch.sum(centered**3)
    m4 = torch.sum(centered**4)
    return count, mean, m2, m3, m4


def _combine_moments(
    n1: torch.Tensor,
    mean1: torch.Tensor,
    m2_1: torch.Tensor,
    m3_1: torch.Tensor,
    m4_1: torch.Tensor,
    n2: torch.Tensor,
    mean2: torch.Tensor,
    m2_2: torch.Tensor,
    m3_2: torch.Tensor,
    m4_2: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    if n1.item() == 0:
        return n2, mean2, m2_2, m3_2, m4_2
    if n2.item() == 0:
        return n1, mean1, m2_1, m3_1, m4_1

    n = n1 + n2
    delta = mean2 - mean1
    delta2 = delta * delta
    delta3 = delta2 * delta
    delta4 = delta2 * delta2
    n1n2 = n1 * n2

    mean = mean1 + delta * n2 / n
    m2 = m2_1 + m2_2 + delta2 * n1n2 / n
    m3 = (
        m3_1
        + m3_2
        + delta3 * n1n2 * (n1 - n2) / (n * n)
        + 3.0 * delta * (n1 * m2_2 - n2 * m2_1) / n
    )
    m4 = (
        m4_1
        + m4_2
        + delta4 * n1n2 * (n1 * n1 - n1n2 + n2 * n2) / (n * n * n)
        + 6.0 * delta2 * (n1 * n1 * m2_2 + n2 * n2 * m2_1) / (n * n)
        + 4.0 * delta * (n1 * m3_2 - n2 * m3_1) / n
    )
    return n, mean, m2, m3, m4


def _init_state() -> Dict[str, torch.Tensor]:
    return {
        "count": torch.tensor(0.0, dtype=torch.float64),
        "mean": torch.tensor(0.0, dtype=torch.float64),
        "m2": torch.tensor(0.0, dtype=torch.float64),
        "m3": torch.tensor(0.0, dtype=torch.float64),
        "m4": torch.tensor(0.0, dtype=torch.float64),
        "min": torch.tensor(float("inf"), dtype=torch.float64),
        "max": torch.tensor(float("-inf"), dtype=torch.float64),
        "count_pos": torch.tensor(0.0, dtype=torch.float64),
        "count_neg": torch.tensor(0.0, dtype=torch.float64),
        "count_zero": torch.tensor(0.0, dtype=torch.float64),
    }


def _update_state(state: Dict[str, torch.Tensor], values: torch.Tensor) -> None:
    values = values.to(dtype=torch.float64)
    if values.numel() == 0:
        return
    state["min"] = torch.minimum(state["min"], values.min())
    state["max"] = torch.maximum(state["max"], values.max())
    state["count_pos"] += torch.sum(values > 0)
    state["count_neg"] += torch.sum(values < 0)
    state["count_zero"] += torch.sum(values == 0)

    n2, mean2, m2_2, m3_2, m4_2 = _compute_moments(values)
    n1, mean1, m2_1, m3_1, m4_1 = (
        state["count"],
        state["mean"],
        state["m2"],
        state["m3"],
        state["m4"],
    )
    n, mean, m2, m3, m4 = _combine_moments(
        n1, mean1, m2_1, m3_1, m4_1, n2, mean2, m2_2, m3_2, m4_2
    )
    state["count"] = n
    state["mean"] = mean
    state["m2"] = m2
    state["m3"] = m3
    state["m4"] = m4


def _finalize_state(state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    count = state["count"].item()
    if count == 0:
        return {}
    mean = state["mean"].item()
    var = (state["m2"] / state["count"]).item()
    std = float(np.sqrt(max(var, 0.0)))
    m2 = state["m2"].item()
    m3 = state["m3"].item()
    m4 = state["m4"].item()
    skew = float(np.sqrt(count) * m3 / (m2**1.5)) if m2 > 0 else 0.0
    kurt = float(count * m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0
    return {
        "count": float(count),
        "mean": float(mean),
        "std": std,
        "min": float(state["min"].item()),
        "max": float(state["max"].item()),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "fraction_positive": float(state["count_pos"].item() / count),
        "fraction_negative": float(state["count_neg"].item() / count),
        "fraction_zero": float(state["count_zero"].item() / count),
    }


def _sample_from_batch(
    density: torch.Tensor,
    susc: torch.Tensor,
    prob: float,
    density_samples: list,
    susc_samples: list,
) -> None:
    if prob <= 0:
        return
    for values, bucket in ((density, density_samples), (susc, susc_samples)):
        mask = torch.rand(values.numel()) < prob
        if torch.any(mask):
            bucket.append(values[mask].cpu().numpy())


def _finalize_samples(
    density_samples: list,
    susc_samples: list,
    sample_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    density_concat = (
        np.concatenate(density_samples, axis=0) if density_samples else np.array([])
    )
    susc_concat = np.concatenate(susc_samples, axis=0) if susc_samples else np.array([])
    if sample_size > 0:
        rng = np.random.default_rng(seed)
        if density_concat.size > sample_size:
            density_concat = rng.choice(density_concat, size=sample_size, replace=False)
        if susc_concat.size > sample_size:
            susc_concat = rng.choice(susc_concat, size=sample_size, replace=False)
    return density_concat, susc_concat


def _percentiles(values: np.ndarray, probs: Iterable[float]) -> Dict[str, float]:
    if values.size == 0:
        return {}
    quantiles = np.quantile(values, probs, method="linear")
    return {f"p{int(p * 1000):03d}": float(q) for p, q in zip(probs, quantiles)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset distribution stats.")
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=8)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stats_path", type=str, default="data_stats.json")
    parser.add_argument("--sample_size", type=int, default=200_000)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_quantiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = JointDensitySuscDataset(args.zarr_path)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
    )

    density_state = _init_state()
    susc_state = _init_state()
    log_density_state = _init_state()
    log_susc_state = _init_state()

    volume_shape = None
    total_count = None
    density_samples = []
    susc_samples = []
    sample_prob = 0.0
    for batch in tqdm(loader, desc="Streaming moments"):
        if volume_shape is None:
            volume_shape = tuple(batch.shape[2:])
            if not args.skip_quantiles and args.sample_size > 0:
                total_count = int(len(dataset) * np.prod(volume_shape))
                sample_prob = min(1.0, args.sample_size / float(total_count))
        density = batch[:, 0].reshape(-1)
        susc = batch[:, 1].reshape(-1)
        _update_state(density_state, density)
        _update_state(susc_state, susc)
        log_density = torch.log1p(torch.clamp(density, min=0))
        log_susc = torch.log1p(torch.clamp(susc, min=0))
        _update_state(log_density_state, log_density)
        _update_state(log_susc_state, log_susc)
        if not args.skip_quantiles:
            _sample_from_batch(
                density,
                susc,
                sample_prob,
                density_samples,
                susc_samples,
            )

    density_stats = _finalize_state(density_state)
    susc_stats = _finalize_state(susc_state)
    log_density_stats = _finalize_state(log_density_state)
    log_susc_stats = _finalize_state(log_susc_state)

    output = {
        "volume_shape": volume_shape,
        "channels": {
            "density": {
                **density_stats,
                "log1p": log_density_stats,
            },
            "susceptibility": {
                **susc_stats,
                "log1p": log_susc_stats,
            },
        },
    }

    if not args.skip_quantiles:
        density_sample, susc_sample = _finalize_samples(
            density_samples,
            susc_samples,
            args.sample_size,
            args.seed,
        )
        output["channels"]["density"]["percentiles"] = _percentiles(
            density_sample, args.quantiles
        )
        output["channels"]["density"]["log1p"]["percentiles"] = _percentiles(
            np.log1p(density_sample), args.quantiles
        )
        output["channels"]["susceptibility"]["percentiles"] = _percentiles(
            susc_sample, args.quantiles
        )
        output["channels"]["susceptibility"]["log1p"]["percentiles"] = _percentiles(
            np.log1p(susc_sample), args.quantiles
        )

    stats_path = Path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
