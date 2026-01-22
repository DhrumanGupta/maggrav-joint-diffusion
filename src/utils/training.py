"""Training utilities for accelerator setup, EMA, and logging."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed

logger = logging.getLogger(__name__)


def create_accelerator(
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    mixed_precision: str = "bf16",
) -> Accelerator:
    """
    Create and configure an Accelerator instance with standard settings.

    Args:
        gradient_accumulation_steps: Number of gradient accumulation steps.
        seed: Random seed for reproducibility.
        mixed_precision: Mixed precision mode ("bf16", "fp16", or "no").

    Returns:
        Configured Accelerator instance.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    set_seed(seed)
    return accelerator


def create_ema(model: nn.Module, decay: float = 0.9999):
    """
    Create an EMA (Exponential Moving Average) model wrapper.

    Args:
        model: The model to track with EMA.
        decay: EMA decay rate (default 0.9999).

    Returns:
        EMAModel instance from diffusers.
    """
    from diffusers.training_utils import EMAModel

    return EMAModel(model.parameters(), decay=decay)


@contextmanager
def ema_eval_context(
    ema_model, model_parameters: Iterator[nn.Parameter]
) -> Iterator[None]:
    """
    Context manager for temporarily using EMA weights during evaluation.

    Stores original weights, copies EMA weights to model, yields control,
    then restores original weights.

    Args:
        ema_model: EMAModel instance.
        model_parameters: Iterator over model parameters.

    Example:
        with ema_eval_context(ema_model, model.parameters()):
            samples = sample_function(model, ...)
    """
    ema_model.store(model_parameters)
    ema_model.copy_to(model_parameters)
    try:
        yield
    finally:
        ema_model.restore(model_parameters)


def setup_wandb(
    project: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
):
    """
    Initialize Weights & Biases logging (only on main process).

    Args:
        project: W&B project name.
        config: Configuration dictionary to log.
        run_name: Optional name for the run.
        accelerator: Accelerator instance (if provided, only main process initializes).

    Returns:
        W&B run object or None if not main process.
    """
    import wandb

    # Only initialize on main process
    if accelerator is not None and not accelerator.is_main_process:
        return None

    init_kwargs = {
        "project": project,
        "config": config,
    }
    if run_name:
        init_kwargs["name"] = run_name

    return wandb.init(**init_kwargs)


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    ema_model=None,
    **extra_fields,
) -> None:
    """
    Save a training checkpoint (only on main process).

    Args:
        accelerator: Accelerator instance.
        model: Model to save (will be unwrapped).
        optimizer: Optimizer to save.
        checkpoint_path: Path to save the checkpoint.
        ema_model: Optional EMA model to save.
        **extra_fields: Additional fields to include in the checkpoint.
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    unwrapped_model = accelerator.unwrap_model(model)
    checkpoint = {
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **extra_fields,
    }
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved checkpoint: %s", checkpoint_path)
