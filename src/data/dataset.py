"""
PyTorch Dataset for EEG Motor Imagery Data
===========================================

Custom Dataset class для загрузки и аугментации EEG данных.

Автор: Temur Turayev
TashPMI, 2024
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


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
            print(f"📊 EEGDataset loaded:")
            print(f"   Split: {split}")
            print(f"   Samples: {len(self)}")
            print(f"   Shape: {self.data.shape}")
            print(f"   Classes: {len(np.unique(self.labels))}")
            print(f"   Augmentation: {self.augment}")

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
            1. Time shifting: Shift signal in time
            2. Amplitude scaling: Scale amplitude
            3. Additive noise: Add small Gaussian noise

        Medical Context:
            - EEG signals vary naturally between trials
            - Augmentation helps model generalize to this variability
            - Important: Don't augment so much that signal becomes unrealistic
        """
        augmented = signal.copy()

        # 1. Time shifting (50% probability)
        if np.random.rand() < 0.5:
            max_shift = int(0.1 * signal.shape[1])  # Max 10% shift
            shift = np.random.randint(-max_shift, max_shift)
            augmented = np.roll(augmented, shift, axis=1)

        # 2. Amplitude scaling (50% probability)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)  # ±10% scaling
            augmented = augmented * scale

        # 3. Additive noise (30% probability)
        if np.random.rand() < 0.3:
            noise_level = 0.01 * np.std(augmented)  # 1% of signal std
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise

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
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders.

    Args:
        data_path: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (0 = main thread)
        augment_train: Apply augmentation to training data
        verbose: Print information

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data

    DataLoader Benefits:
        - Automatic batching
        - Shuffling (for training)
        - Parallel loading (if num_workers > 0)
        - Memory efficient (loads data on demand)

    Usage:
        >>> train_loader, test_loader = create_data_loaders('data/processed/')
        >>> for signals, labels in train_loader:
        ...     # signals: [batch_size, 1, n_channels, n_samples]
        ...     # labels: [batch_size]
        ...     predictions = model(signals)
    """
    # Create datasets
    train_dataset = EEGDataset(
        data_path=data_path,
        split='train',
        augment=augment_train,
        verbose=verbose
    )

    test_dataset = EEGDataset(
        data_path=data_path,
        split='test',
        augment=False,  # Never augment test data!
        verbose=verbose
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False  # Faster GPU transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if verbose:
        print(f"\n📦 DataLoaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        print(f"   Batch size: {batch_size}")

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
        class_name = ['Left Hand', 'Right Hand', 'Feet', 'Tongue'][i]
        print(f"   {class_name}: {weight:.4f}")

    print("\n✅ Dataset test passed!")
    print("="*60)


if __name__ == "__main__":
    main()
