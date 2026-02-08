"""
Model Utilities
================

Helper functions для работы с моделями EEGNet.

Автор: Temur Turayev
TashPMI, 2024
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, Optional


def get_device(preference: str = "auto") -> torch.device:
    """
    Detect best available device with robust MPS fallback.

    Args:
        preference: Device preference. Use "auto" for automatic detection,
                    or specify "cuda", "mps", "cpu" directly.

    Returns:
        torch.device for the best available backend.
    """
    if preference != "auto":
        return torch.device(preference)
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str,
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        loss: Training loss
        accuracy: Validation accuracy
        filepath: Path to save checkpoint
        model_config: Optional dict of model constructor parameters
        training_config: Optional dict of training hyperparameters
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    if model_config is not None:
        checkpoint['model_config'] = model_config
    if training_config is not None:
        checkpoint['training_config'] = training_config
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to map tensors to

    Returns:
        Dictionary with checkpoint info

    Note:
        Uses ``weights_only=False`` so that non-tensor metadata stored in
        the checkpoint (e.g. model_config, training_config dicts) can be
        restored. Only load checkpoints that you trust.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
