"""
Training Configuration
=======================

Hyperparameters и настройки для обучения EEGNet.

Автор: Temur Turayev
TashPMI, 2024
"""

from dataclasses import dataclass, fields
from pathlib import Path

from src.constants import (
    N_CLASSES, N_CHANNELS, N_SAMPLES,
    LOWCUT, HIGHCUT, SAMPLING_RATE,
)


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
    n_classes: int = N_CLASSES
    n_channels: int = N_CHANNELS
    n_samples: int = N_SAMPLES

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

    gradient_clip_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "plateau"  # "plateau" or "cosine"
    cosine_T_0: int = 50

    max_norm_constraint: float = 0.25

    # Data augmentation
    augment_train: bool = True

    # Validation split
    val_size: float = 0.15

    # Experiment
    experiment_name: str = "default"

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

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML config file.

        Returns:
            TrainingConfig populated from the YAML values.
        """
        import yaml

        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f) or {}

        valid_fields = {field.name for field in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


# Default configuration
DEFAULT_CONFIG = TrainingConfig()
