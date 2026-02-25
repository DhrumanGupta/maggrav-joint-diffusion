"""
Compute distribution statistics for density + susceptibility volumes.

Usage:
    python -m src.data.compute_data_stats --zarr_path /path/to/zarr
    torchrun --standalone --nproc_per_node=8 -m src.data.compute_data_stats --zarr_path /path/to/zarr --device cuda
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..utils.datasets import JointDensitySuscDataset

logger = logging.getLogger(__name__)

MOMENT_DTYPE = torch.float32
REGISTER_DTYPE = torch.float64
DEFAULT_REPORT_INTERVAL_SECONDS = 10.0


def _configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def _compute_moments(values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    values = values.to(dtype=MOMENT_DTYPE)
    count = torch.tensor(values.numel(), dtype=REGISTER_DTYPE, device=values.device)
    mean = values.mean()
    centered = values - mean
    m2 = torch.sum(centered**2)
    m3 = torch.sum(centered**3)
    m4 = torch.sum(centered**4)
    return (
        count,
        mean.to(dtype=REGISTER_DTYPE),
        m2.to(dtype=REGISTER_DTYPE),
        m3.to(dtype=REGISTER_DTYPE),
        m4.to(dtype=REGISTER_DTYPE),
    )


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


def _init_state(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "count": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "mean": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "m2": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "m3": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "m4": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "min": torch.tensor(float("inf"), dtype=REGISTER_DTYPE, device=device),
        "max": torch.tensor(float("-inf"), dtype=REGISTER_DTYPE, device=device),
        "count_pos": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "count_neg": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
        "count_zero": torch.tensor(0.0, dtype=REGISTER_DTYPE, device=device),
    }


def _state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _merge_state(
    aggregate: Dict[str, torch.Tensor], update: Dict[str, torch.Tensor]
) -> None:
    aggregate["min"] = torch.minimum(aggregate["min"], update["min"])
    aggregate["max"] = torch.maximum(aggregate["max"], update["max"])
    aggregate["count_pos"] += update["count_pos"]
    aggregate["count_neg"] += update["count_neg"]
    aggregate["count_zero"] += update["count_zero"]

    n, mean, m2, m3, m4 = _combine_moments(
        aggregate["count"],
        aggregate["mean"],
        aggregate["m2"],
        aggregate["m3"],
        aggregate["m4"],
        update["count"],
        update["mean"],
        update["m2"],
        update["m3"],
        update["m4"],
    )
    aggregate["count"] = n
    aggregate["mean"] = mean
    aggregate["m2"] = m2
    aggregate["m3"] = m3
    aggregate["m4"] = m4


def _update_state(state: Dict[str, torch.Tensor], values: torch.Tensor) -> None:
    values = values.to(dtype=MOMENT_DTYPE)
    if values.numel() == 0:
        return
    state["min"] = torch.minimum(state["min"], values.min().to(dtype=REGISTER_DTYPE))
    state["max"] = torch.maximum(state["max"], values.max().to(dtype=REGISTER_DTYPE))
    state["count_pos"] += torch.sum(values > 0).to(dtype=REGISTER_DTYPE)
    state["count_neg"] += torch.sum(values < 0).to(dtype=REGISTER_DTYPE)
    state["count_zero"] += torch.sum(values == 0).to(dtype=REGISTER_DTYPE)

    n2, mean2, m2_2, m3_2, m4_2 = _compute_moments(values)
    n, mean, m2, m3, m4 = _combine_moments(
        state["count"],
        state["mean"],
        state["m2"],
        state["m3"],
        state["m4"],
        n2,
        mean2,
        m2_2,
        m3_2,
        m4_2,
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
        mask = torch.rand(values.numel(), device=values.device) < prob
        if torch.any(mask):
            bucket.append(values[mask].detach().cpu().numpy())


def _concat_samples(samples: list) -> np.ndarray:
    if not samples:
        return np.array([], dtype=np.float32)
    return np.concatenate(samples, axis=0)


def _finalize_samples(
    density_samples: list,
    susc_samples: list,
    sample_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    density_concat = _concat_samples(density_samples)
    susc_concat = _concat_samples(susc_samples)
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


def _resolve_launch_context(
    args: argparse.Namespace,
) -> Tuple[str, int, int, int, bool]:
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = env_world_size > 1

    if requested == "cpu":
        if args.num_gpus not in (0, 1):
            logger.warning(
                "Ignoring --num_gpus=%d because device is CPU.", args.num_gpus
            )
        if distributed:
            raise ValueError("Distributed mode with device=cpu is not supported here.")
        return "cpu", 0, 0, 1, False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    available = torch.cuda.device_count()

    if distributed:
        world_size = env_world_size
        if args.num_gpus > 0 and args.num_gpus != world_size:
            raise ValueError(
                f"--num_gpus={args.num_gpus} does not match WORLD_SIZE={world_size}."
            )
        if env_local_rank >= available:
            raise ValueError(
                f"LOCAL_RANK={env_local_rank} but only {available} CUDA devices are visible."
            )
        return "cuda", env_rank, env_local_rank, world_size, True

    world_size = available if args.num_gpus <= 0 else args.num_gpus
    if world_size > available:
        raise ValueError(
            f"Requested {world_size} GPUs but only {available} CUDA devices are available."
        )
    if world_size > 1:
        raise RuntimeError(
            "For multi-GPU, launch with torchrun, e.g. "
            "`torchrun --standalone --nproc_per_node=8 -m src.data.compute_data_stats ...`"
        )
    return "cuda", 0, 0, 1, False


def _build_rank_dataset(
    args: argparse.Namespace, rank: int, world_size: int
) -> Tuple[Subset, int]:
    dataset = JointDensitySuscDataset(args.zarr_path)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    global_total_samples = len(dataset)
    if world_size > 1:
        dataset = Subset(dataset, range(rank, global_total_samples, world_size))
    return dataset, global_total_samples


def _compute_rank(
    rank: int,
    local_rank: int,
    world_size: int,
    args: argparse.Namespace,
    device_type: str,
) -> Dict:
    run_start = time.perf_counter()
    show_rank_logs = args.log_all_ranks or rank == 0

    if device_type == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(args.seed + rank)
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    dataset, global_total_samples = _build_rank_dataset(args, rank, world_size)
    local_total_samples = len(dataset)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device_type == "cuda",
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": (
            args.persistent_workers if args.num_workers > 0 else False
        ),
    }
    loader = DataLoader(**loader_kwargs)

    density_state = _init_state(device)
    susc_state = _init_state(device)
    log_density_state = _init_state(device)
    log_susc_state = _init_state(device)

    volume_shape = None
    total_voxels = 0
    processed_samples = 0
    num_batches = 0

    density_samples = []
    susc_samples = []
    sample_prob = 0.0

    loader_seconds_total = 0.0
    compute_seconds_total = 0.0
    loader_seconds_report = 0.0
    compute_seconds_report = 0.0
    samples_report = 0
    voxels_report = 0
    batches_report = 0

    loop_marker = time.perf_counter()
    last_report_time = run_start
    progress_desc = (
        "Streaming moments" if world_size == 1 else f"Streaming moments r{rank}"
    )

    for batch in tqdm(loader, desc=progress_desc, disable=not show_rank_logs):
        batch_ready = time.perf_counter()
        loader_delta = batch_ready - loop_marker
        loader_seconds_total += loader_delta
        loader_seconds_report += loader_delta
        compute_start = batch_ready

        if device_type == "cuda":
            batch = batch.to(device=device, non_blocking=True)

        if volume_shape is None:
            volume_shape = tuple(batch.shape[2:])
            if not args.skip_quantiles and args.sample_size > 0:
                global_total_voxels = int(global_total_samples * np.prod(volume_shape))
                sample_prob = min(1.0, args.sample_size / float(global_total_voxels))

        batch_samples = int(batch.shape[0])
        batch_voxels = int(batch.shape[0] * np.prod(batch.shape[2:]))
        total_voxels += batch_voxels
        processed_samples += batch_samples
        samples_report += batch_samples
        voxels_report += batch_voxels
        num_batches += 1
        batches_report += 1

        density = batch[:, 0].reshape(-1)
        susc = batch[:, 1].reshape(-1)

        _update_state(density_state, density)
        _update_state(susc_state, susc)
        _update_state(log_density_state, torch.log1p(torch.clamp(density, min=0)))
        _update_state(log_susc_state, torch.log1p(torch.clamp(susc, min=0)))

        if not args.skip_quantiles:
            _sample_from_batch(
                density,
                susc,
                sample_prob,
                density_samples,
                susc_samples,
            )

        loop_marker = time.perf_counter()
        compute_delta = loop_marker - compute_start
        compute_seconds_total += compute_delta
        compute_seconds_report += compute_delta

        if (
            show_rank_logs
            and args.report_interval_seconds > 0
            and loop_marker - last_report_time >= args.report_interval_seconds
        ):
            elapsed = loop_marker - run_start
            interval_elapsed = loop_marker - last_report_time
            samples_per_second_window = (
                float(samples_report / interval_elapsed)
                if interval_elapsed > 0
                else 0.0
            )
            voxels_per_second_window = (
                float(voxels_report / interval_elapsed) if interval_elapsed > 0 else 0.0
            )
            logger.info(
                "\n[rank %d timing] elapsed=%.1fs interval=%.1fs "
                "batches=%d(+%d) samples=%d/%d(+%d) "
                "fetch=%.1fs compute=%.1fs compute_total=%.1fs "
                "samples/s=%.1f voxels/s/ch=%.1f",
                rank,
                elapsed,
                interval_elapsed,
                num_batches,
                batches_report,
                processed_samples,
                local_total_samples,
                samples_report,
                loader_seconds_report,
                compute_seconds_report,
                compute_seconds_total,
                samples_per_second_window,
                voxels_per_second_window,
            )
            last_report_time = loop_marker
            loader_seconds_report = 0.0
            compute_seconds_report = 0.0
            samples_report = 0
            voxels_report = 0
            batches_report = 0

    if device_type == "cuda":
        torch.cuda.synchronize(device)

    total_seconds = time.perf_counter() - run_start
    return {
        "rank": rank,
        "volume_shape": volume_shape,
        "density_state": _state_to_cpu(density_state),
        "susc_state": _state_to_cpu(susc_state),
        "log_density_state": _state_to_cpu(log_density_state),
        "log_susc_state": _state_to_cpu(log_susc_state),
        "density_sample": _concat_samples(density_samples),
        "susc_sample": _concat_samples(susc_samples),
        "timing": {
            "rank": rank,
            "num_batches": num_batches,
            "num_samples": processed_samples,
            "num_voxels_per_channel": total_voxels,
            "loader_seconds": float(loader_seconds_total),
            "compute_seconds": float(compute_seconds_total),
            "total_seconds": float(total_seconds),
            "samples_per_second": (
                float(processed_samples / total_seconds) if total_seconds > 0 else 0.0
            ),
            "voxels_per_second_per_channel": (
                float(total_voxels / total_seconds) if total_seconds > 0 else 0.0
            ),
        },
    }


def _run_rank_and_collect(
    args: argparse.Namespace,
    device_type: str,
    rank: int,
    local_rank: int,
    world_size: int,
    distributed: bool,
) -> List[Dict]:
    result = _compute_rank(rank, local_rank, world_size, args, device_type)
    if not distributed:
        return [result]

    gathered: List[Dict] = [None] * world_size  # type: ignore[assignment]
    dist.all_gather_object(gathered, result)
    return gathered if rank == 0 else []


def _aggregate_results(results: List[Dict], args: argparse.Namespace) -> Dict:
    cpu_device = torch.device("cpu")
    density_state = _init_state(cpu_device)
    susc_state = _init_state(cpu_device)
    log_density_state = _init_state(cpu_device)
    log_susc_state = _init_state(cpu_device)

    volume_shape = None
    total_voxels = 0
    total_samples = 0
    total_batches = 0
    density_samples = []
    susc_samples = []
    timing_rows = []

    for result in results:
        if volume_shape is None:
            volume_shape = result["volume_shape"]
        elif result["volume_shape"] is not None and tuple(
            result["volume_shape"]
        ) != tuple(volume_shape):
            raise ValueError(
                f"Mismatched volume shapes across ranks: {volume_shape} vs {result['volume_shape']}"
            )

        _merge_state(density_state, result["density_state"])
        _merge_state(susc_state, result["susc_state"])
        _merge_state(log_density_state, result["log_density_state"])
        _merge_state(log_susc_state, result["log_susc_state"])

        density_samples.append(result["density_sample"])
        susc_samples.append(result["susc_sample"])

        timing = result["timing"]
        timing_rows.append(timing)
        total_voxels += int(timing["num_voxels_per_channel"])
        total_samples += int(timing["num_samples"])
        total_batches += int(timing["num_batches"])

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

    output["timing"] = {
        "num_batches": total_batches,
        "num_samples": total_samples,
        "num_voxels_per_channel": total_voxels,
        "loader_seconds": float(sum(row["loader_seconds"] for row in timing_rows)),
        "compute_seconds": float(sum(row["compute_seconds"] for row in timing_rows)),
        "worker_wall_seconds": float(max(row["total_seconds"] for row in timing_rows)),
        "per_rank": timing_rows,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset distribution stats.")
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Execution device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="In single-process mode: number of GPUs (0=all visible, requires <=1). "
        "In torchrun mode: optional check that matches WORLD_SIZE.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per rank (per GPU in multi-GPU mode).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers per rank (per GPU in multi-GPU mode).",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--max_samples", type=int, default=200000)
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
    parser.add_argument(
        "--log_all_ranks",
        action="store_true",
        help="If set, print periodic timing logs from every rank (default is rank 0 only).",
    )
    parser.add_argument(
        "--report_interval_seconds",
        type=float,
        default=DEFAULT_REPORT_INTERVAL_SECONDS,
        help="Print timing/throughput progress every N seconds. Set <=0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging()

    device_type, rank, local_rank, world_size, distributed = _resolve_launch_context(
        args
    )

    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")

    try:
        if args.log_all_ranks or rank == 0:
            logger.info(
                "Starting stats pass: device=%s world_size=%d rank=%d local_rank=%d "
                "batch_size_per_rank=%d num_workers_per_rank=%d",
                device_type,
                world_size,
                rank,
                local_rank,
                args.batch_size,
                args.num_workers,
            )

        run_start = time.perf_counter()
        results = _run_rank_and_collect(
            args=args,
            device_type=device_type,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed=distributed,
        )
        post_start = time.perf_counter()

        if rank != 0:
            return

        output = _aggregate_results(results, args)
        postprocess_seconds = time.perf_counter() - post_start
        total_seconds = time.perf_counter() - run_start

        total_samples = output["timing"]["num_samples"]
        total_voxels = output["timing"]["num_voxels_per_channel"]
        output["timing"]["postprocess_seconds"] = float(postprocess_seconds)
        output["timing"]["total_seconds"] = float(total_seconds)
        output["timing"]["device"] = device_type
        output["timing"]["world_size"] = world_size
        output["timing"]["samples_per_second"] = (
            float(total_samples / total_seconds) if total_seconds > 0 else 0.0
        )
        output["timing"]["voxels_per_second_per_channel"] = (
            float(total_voxels / total_seconds) if total_seconds > 0 else 0.0
        )

        stats_path = Path(args.stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as f:
            json.dump(output, f, indent=2)

        logger.info("Wrote stats to %s", stats_path)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
