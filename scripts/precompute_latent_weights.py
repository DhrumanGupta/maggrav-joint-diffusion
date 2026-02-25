"""Precompute density-aware latent weights for a latent zarr store."""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.datasets import LatentZarrDataset
from src.utils.latent_weighting import LatentDensityKNNWeighter

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

zarr_path = "./zarrs/encoded-latents-attention-sweep-3.zarr"

weighter = LatentDensityKNNWeighter(
    mode="recompute",  # force compute + overwrite cache
    cache_path=None,  # default: next to zarr, <stem>.density_weights.pt
    proj_dim=4096,
    proj_seed=0,
    k=30,
    alpha=1.5,
    w_min=0.02,
    w_max=5.0,
    distance="cosine",
    normalize_embeddings=True,
    index_factory="IVF16384,PQ64x8",
    nprobe=64,
    train_size=200000,
    use_gpu=True,
    faiss_num_gpus=0,  # 0 = use all visible GPUs (FAISS)
    projection_feature_chunk=16384,
    projection_backend="torch_countsketch",
)

# Construction triggers filter/weighter build_plan and cache write
ds = LatentZarrDataset(
    zarr_path=zarr_path,
    use_logvar=False,
    sample_filter=weighter,
)

print(f"Done. Dataset size: {len(ds)}")
