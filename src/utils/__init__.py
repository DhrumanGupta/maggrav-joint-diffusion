"""Common utilities for the maggrav project."""

from .checkpoint import clean_state_dict, load_checkpoint
from .dataset_filters import (
    DatasetFilter,
    DatasetFilterContext,
    DatasetSamplingPlan,
    PerSampleDatasetFilter,
)
from .datasets import JointDensitySuscDataset, LatentFilterBatch, LatentZarrDataset
from .latent_weighted_filtering import LatentWeightedFiltering
from .latent_weighting import LatentDensityKNNWeighter
from .masking import create_padding_mask
from .stats import get_effective_std, load_stats, reshape_stats_for_broadcast
from .training import create_accelerator, create_ema, ema_eval_context, setup_wandb

__all__ = [
    # checkpoint
    "load_checkpoint",
    "clean_state_dict",
    # dataset filters
    "DatasetFilterContext",
    "DatasetFilter",
    "DatasetSamplingPlan",
    "PerSampleDatasetFilter",
    # datasets
    "JointDensitySuscDataset",
    "LatentFilterBatch",
    "LatentZarrDataset",
    "LatentDensityKNNWeighter",
    "LatentWeightedFiltering",
    # masking
    "create_padding_mask",
    # stats
    "get_effective_std",
    "load_stats",
    "reshape_stats_for_broadcast",
    # training
    "create_accelerator",
    "create_ema",
    "ema_eval_context",
    "setup_wandb",
]
