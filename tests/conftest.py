"""
Shared test fixtures for NeuroHand.
"""

import numpy as np
import pytest
import torch

from src.constants import N_CHANNELS, N_CLASSES, N_SAMPLES


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_eeg(rng):
    """Single EEG trial: [n_channels, n_samples]."""
    return rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)


@pytest.fixture
def batch_eeg(rng):
    """Batch of EEG trials: [batch, n_channels, n_samples]."""
    return rng.standard_normal((8, N_CHANNELS, N_SAMPLES)).astype(np.float32)


@pytest.fixture
def sample_labels(rng):
    """Random labels for a batch of 8."""
    return rng.integers(0, N_CLASSES, size=8)


@pytest.fixture
def model_input(rng):
    """Model-ready tensor: [batch, 1, n_channels, n_samples]."""
    data = rng.standard_normal((4, 1, N_CHANNELS, N_SAMPLES)).astype(np.float32)
    return torch.from_numpy(data)


@pytest.fixture
def device():
    """Best available device for testing."""
    from src.models.utils import get_device
    return get_device("cpu")


@pytest.fixture
def tmp_data_dir(tmp_path, rng):
    """Create temporary directory with synthetic train/val/test data."""
    n_train, n_val, n_test = 40, 10, 10

    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        data = rng.standard_normal((n, N_CHANNELS, N_SAMPLES)).astype(np.float32)
        # Ensure all classes are represented (cycle through 0..N_CLASSES-1)
        labels = np.array([i % N_CLASSES for i in range(n)], dtype=np.int64)
        np.save(tmp_path / f"{split}_data.npy", data)
        np.save(tmp_path / f"{split}_labels.npy", labels)

    return tmp_path
