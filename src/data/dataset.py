"""
PyTorch Dataset for EEG Motor Imagery Data
===========================================

Custom Dataset class для загрузки и аугментации EEG данных.

Автор: Temur Turayev
TashPMI, 2024
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

from src.constants import CLASS_NAMES, N_CLASSES
from src.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG motor imagery data.

    Загружает preprocessed EEG данные и возвращает их в формате PyTorch tensors.
    Поддерживает data augmentation для улучшения генерализации модели.
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        augment: bool = False,
        verbose: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to processed data directory
            split: 'train' or 'test'
            transform: Optional transform to apply to data
            augment: Apply data augmentation (only for training)
            verbose: Print loading information

        Raises:
            FileNotFoundError: If data files don't exist
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.augment = augment and (split == 'train')  # Only augment training data
        self.verbose = verbose

        # Load data
        self._load_data()

        if self.verbose:
            logger.info("EEGDataset loaded: split=%s, samples=%d, shape=%s, classes=%d, augmentation=%s",
                         split, len(self), self.data.shape,
                         len(np.unique(self.labels)), self.augment)

    def _load_data(self):
        """Load preprocessed data from disk."""
        # Construct file paths
        data_file = self.data_path / f"{self.split}_data.npy"
        labels_file = self.data_path / f"{self.split}_labels.npy"

        # Check if files exist
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please run: python src/data/preprocessing.py --create_split"
            )

        # Load data
        self.data = np.load(data_file)  # [n_trials, n_channels, n_samples]
        self.labels = np.load(labels_file)  # [n_trials]

        # Load dataset info if available
        info_file = self.data_path / "dataset_info.pkl"
        if info_file.exists():
            with open(info_file, 'rb') as f:
                self.info = pickle.load(f)
        else:
            self.info = None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample from dataset.

        Args:
            idx: Sample index

        Returns:
            signal: EEG signal [1, n_channels, n_samples]
            label: Class label (0, 1, 2, or 3)

        Data Format:
            - Input to EEGNet expects shape: [batch, 1, channels, samples]
            - We add channel dimension (1) to indicate single time series
        """
        # Get data
        signal = self.data[idx]  # [n_channels, n_samples]
        label = self.labels[idx]

        # Apply data augmentation if enabled
        if self.augment:
            signal = self._augment_signal(signal)

        # Convert to PyTorch tensors
        signal = torch.from_numpy(signal).float()
        label = torch.tensor(label, dtype=torch.long)

        # Add channel dimension: [n_channels, n_samples] -> [1, n_channels, n_samples]
        signal = signal.unsqueeze(0)

        # Apply custom transform if provided
        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label

    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to EEG signal.

        Args:
            signal: EEG signal [n_channels, n_samples]

        Returns:
            Augmented signal

        Augmentation Techniques:
            1. Time shifting with zero-padding (no wrap-around)
            2. Amplitude scaling
            3. Additive noise
            4. Channel dropout

        Medical Context:
            - EEG signals vary naturally between trials
            - Augmentation helps model generalize to this variability
            - Important: Don't augment so much that signal becomes unrealistic
        """
        augmented = signal.copy()

        # 1. Time shifting with zero-padding (50% probability)
        if np.random.rand() < 0.5:
            max_shift = int(0.1 * signal.shape[1])  # Max 10% shift
            shift = np.random.randint(-max_shift, max_shift)
            shifted = np.zeros_like(augmented)
            if shift > 0:
                shifted[:, shift:] = augmented[:, :-shift]
            elif shift < 0:
                shifted[:, :shift] = augmented[:, -shift:]
            else:
                shifted = augmented
            augmented = shifted

        # 2. Amplitude scaling (50% probability)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            augmented = augmented * scale

        # 3. Additive noise (30% probability)
        if np.random.rand() < 0.3:
            noise_level = 0.01 * np.std(augmented)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise

        # 4. Channel dropout (20% probability, zero out 1-2 random channels)
        if np.random.rand() < 0.2:
            n_drop = np.random.randint(1, 3)
            drop_channels = np.random.choice(
                augmented.shape[0], size=n_drop, replace=False
            )
            augmented[drop_channels, :] = 0.0

        return augmented

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            Class weights tensor [n_classes]

        Why Class Weights?
            - If classes are imbalanced (e.g., 100 samples of class 0, 200 of class 1)
            - Model will bias towards majority class
            - Weights compensate: weight = 1 / (frequency of class)
        """
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()  # Normalize
        return torch.from_numpy(class_weights).float()


def create_data_loaders(
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    augment_train: bool = True,
    include_val: bool = False,
    verbose: bool = False
) -> Tuple[DataLoader, ...]:
    """
    Create train, test, and optionally validation DataLoaders.

    Args:
        data_path: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (0 = main thread)
        augment_train: Apply augmentation to training data
        include_val: If True, also return a validation DataLoader
        verbose: Print information

    Returns:
        Tuple of (train_loader, test_loader) or
        (train_loader, val_loader, test_loader) when include_val=True

    Usage:
        >>> train_loader, test_loader = create_data_loaders('data/processed/')
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     'data/processed/', include_val=True)
    """
    pin = torch.cuda.is_available()

    train_dataset = EEGDataset(
        data_path=data_path,
        split='train',
        augment=augment_train,
        verbose=verbose
    )

    test_dataset = EEGDataset(
        data_path=data_path,
        split='test',
        augment=False,
        verbose=verbose
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    if include_val:
        val_dataset = EEGDataset(
            data_path=data_path,
            split='val',
            augment=False,
            verbose=verbose
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin
        )

    if verbose:
        logger.info("DataLoaders created: train_batches=%d, test_batches=%d, batch_size=%d",
                     len(train_loader), len(test_loader), batch_size)
        if include_val:
            logger.info("Validation batches: %d", len(val_loader))

    if include_val:
        return train_loader, val_loader, test_loader

    return train_loader, test_loader


def main():
    """
    Test dataset loading.

    Usage:
        python dataset.py
    """
    print("="*60)
    print("🧪 TESTING DATASET")
    print("="*60)

    # Test dataset creation
    data_path = PROJECT_ROOT / "data" / "processed"

    if not (data_path / "train_data.npy").exists():
        print("❌ Processed data not found!")
        print("   Please run: python src/data/preprocessing.py --create_split")
        return

    # Create datasets
    train_dataset = EEGDataset(
        data_path=str(data_path),
        split='train',
        augment=True,
        verbose=True
    )

    test_dataset = EEGDataset(
        data_path=str(data_path),
        split='test',
        augment=False,
        verbose=True
    )

    # Test loading one sample
    print("\n🔍 Testing sample loading...")
    signal, label = train_dataset[0]
    print(f"   Signal shape: {signal.shape}")
    print(f"   Signal dtype: {signal.dtype}")
    print(f"   Label: {label.item()}")
    print(f"   Signal range: [{signal.min():.3f}, {signal.max():.3f}]")

    # Test DataLoader
    print("\n🔍 Testing DataLoader...")
    train_loader, test_loader = create_data_loaders(
        data_path=str(data_path),
        batch_size=32,
        verbose=True
    )

    # Get one batch
    signals, labels = next(iter(train_loader))
    print(f"   Batch signals shape: {signals.shape}")
    print(f"   Batch labels shape: {labels.shape}")

    # Calculate class weights
    print("\n⚖️  Class weights:")
    class_weights = train_dataset.get_class_weights()
    for i, weight in enumerate(class_weights):
        class_name = CLASS_NAMES[i]
        print(f"   {class_name}: {weight:.4f}")

    print("\n✅ Dataset test passed!")
    print("="*60)


if __name__ == "__main__":
    main()
