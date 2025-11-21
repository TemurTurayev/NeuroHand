"""
Training Configuration
=======================

Hyperparameters и настройки для обучения EEGNet.

Автор: Temur Turayev
TashPMI, 2024
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for training EEGNet.

    Hyperparameter Explanation:
        - epochs: How many times to iterate through entire dataset
        - batch_size: How many samples to process before updating weights
        - learning_rate: Step size for weight updates (smaller = more careful)
        - weight_decay: L2 regularization strength (prevents overfitting)
        - early_stopping_patience: Stop if no improvement after N epochs
    """

    # Data
    data_dir: str = "data/processed"
    n_classes: int = 4
    n_channels: int = 22
    n_samples: int = 1000

    # Model
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    dropout_rate: float = 0.5
    norm_rate: float = 0.25

    # Training
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 50

    # Data augmentation
    augment_train: bool = True

    # Device
    device: str = "auto"  # 'auto', 'cuda', 'mps', or 'cpu'

    # Checkpoints
    save_dir: str = "models/checkpoints"
    save_best_only: bool = True

    # Logging
    log_interval: int = 10  # Print every N batches
    verbose: bool = True

    # Random seed
    random_seed: int = 42

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)


# Default configuration
DEFAULT_CONFIG = TrainingConfig()
