"""
Train BiFlowNet with rectified flow on VAE latents.
"""

import warnings

warnings.filterwarnings("ignore", message="Profiler function")

import argparse
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from ..models.biflownet import BiFlowNet
from ..models.flow import rectified_flow_loss, sample_rectified_flow
from .biflownet_trainer_core import ObjectiveHooks, parse_config_args, run_biflownet_training


def _wrap_flow_model(raw_model: BiFlowNet, _args: argparse.Namespace) -> torch.nn.Module:
    return raw_model


def _flow_per_sample_loss(
    model: torch.nn.Module,
    batch: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    _args: argparse.Namespace,
) -> torch.Tensor:
    return rectified_flow_loss(
        model,
        batch,
        mask=padding_mask,
        return_per_sample=True,
    )


def _flow_sample_latents(
    model: torch.nn.Module,
    local_eval: int,
    latent_shape: Tuple[int, int, int, int],
    device: torch.device,
    padding_mask: Optional[torch.Tensor],
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> torch.Tensor:
    return sample_rectified_flow(
        model,
        num_samples=local_eval,
        latent_shape=latent_shape,
        num_steps=args.num_eval_steps,
        device=device,
        autocast_fn=accelerator.autocast,
        mask=padding_mask,
    )


def _flow_checkpoint_extras(_args: argparse.Namespace) -> Dict[str, Any]:
    return {}


def main() -> None:
    args = parse_config_args(
        description="Train BiFlowNet with rectified flow on latent Zarr.",
        default_config="config/train_flow_biflownet.yaml",
    )
    hooks = ObjectiveHooks(
        wrap_model=_wrap_flow_model,
        compute_per_sample_loss=_flow_per_sample_loss,
        sample_latents=_flow_sample_latents,
        wandb_project="maggrav-biflownet",
        checkpoint_prefix="flow_biflownet_checkpoint",
        checkpoint_extras=_flow_checkpoint_extras,
    )
    run_biflownet_training(args, hooks)


if __name__ == "__main__":
    main()
