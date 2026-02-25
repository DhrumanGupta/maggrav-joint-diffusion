"""
Train BiFlowNet with EDM on VAE latents.
"""

import warnings

warnings.filterwarnings("ignore", message="Profiler function")

import argparse
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from ..models.biflownet import BiFlowNet
from ..models.edm import EDMPrecond, edm_loss, sample_edm_heun
from .biflownet_trainer_core import ObjectiveHooks, parse_config_args, run_biflownet_training


def _wrap_edm_model(raw_model: BiFlowNet, args: argparse.Namespace) -> torch.nn.Module:
    return EDMPrecond(raw_model, sigma_data=args.sigma_data)


def _edm_per_sample_loss(
    model: torch.nn.Module,
    batch: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    return edm_loss(
        model,
        batch,
        mask=padding_mask,
        P_mean=args.P_mean,
        P_std=args.P_std,
        sigma_data=args.sigma_data,
        return_per_sample=True,
    )


def _edm_sample_latents(
    model: torch.nn.Module,
    local_eval: int,
    latent_shape: Tuple[int, int, int, int],
    device: torch.device,
    padding_mask: Optional[torch.Tensor],
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> torch.Tensor:
    return sample_edm_heun(
        model,
        num_samples=local_eval,
        latent_shape=latent_shape,
        num_steps=args.num_eval_steps,
        device=device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        mask=padding_mask,
        autocast_fn=accelerator.autocast,
    )


def _edm_checkpoint_extras(args: argparse.Namespace) -> Dict[str, Any]:
    return {"sigma_data": args.sigma_data}


def main() -> None:
    args = parse_config_args(
        description="Train BiFlowNet with EDM on latent Zarr.",
        default_config="config/train_diffusion_biflownet_edm.yaml",
    )
    hooks = ObjectiveHooks(
        wrap_model=_wrap_edm_model,
        compute_per_sample_loss=_edm_per_sample_loss,
        sample_latents=_edm_sample_latents,
        wandb_project="maggrav-biflownet",
        checkpoint_prefix="edm_biflownet_checkpoint",
        checkpoint_extras=_edm_checkpoint_extras,
    )
    run_biflownet_training(args, hooks)


if __name__ == "__main__":
    main()
