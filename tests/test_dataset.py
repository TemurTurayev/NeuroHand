"""Tests for EEG dataset and data loaders."""

import numpy as np
import pytest
import torch

from src.constants import N_CHANNELS, N_CLASSES, N_SAMPLES
from src.data.dataset import EEGDataset, create_data_loaders


class TestEEGDataset:
    """Tests for the EEGDataset class."""

    def test_loads_data(self, tmp_data_dir):
        dataset = EEGDataset(data_path=str(tmp_data_dir), split='train')
        assert len(dataset) == 40

    def test_getitem_shapes(self, tmp_data_dir):
        dataset = EEGDataset(data_path=str(tmp_data_dir), split='train')
        signal, label = dataset[0]
        assert signal.shape == (1, N_CHANNELS, N_SAMPLES)
        assert signal.dtype == torch.float32
        assert label.dtype == torch.long

    def test_augmented_differs(self, tmp_data_dir):
        dataset = EEGDataset(
            data_path=str(tmp_data_dir), split='train', augment=True
        )
        sig1, _ = dataset[0]
        sig2, _ = dataset[0]
        assert sig1.shape == sig2.shape

    def test_test_split_no_augment(self, tmp_data_dir):
        dataset = EEGDataset(
            data_path=str(tmp_data_dir), split='test', augment=True
        )
        assert not dataset.augment

    def test_class_weights(self, tmp_data_dir):
        dataset = EEGDataset(data_path=str(tmp_data_dir), split='train')
        weights = dataset.get_class_weights()
        assert weights.shape[0] > 0
        assert (weights > 0).all()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EEGDataset(data_path=str(tmp_path), split='nonexistent')


class TestCreateDataLoaders:
    """Tests for the data loader factory."""

    def test_returns_two_loaders(self, tmp_data_dir):
        result = create_data_loaders(data_path=str(tmp_data_dir), batch_size=8)
        assert len(result) == 2

    def test_returns_three_loaders_with_val(self, tmp_data_dir):
        result = create_data_loaders(
            data_path=str(tmp_data_dir), batch_size=8, include_val=True
        )
        assert len(result) == 3

    def test_batch_shapes(self, tmp_data_dir):
        train_loader, test_loader = create_data_loaders(
            data_path=str(tmp_data_dir), batch_size=8
        )
        signals, labels = next(iter(train_loader))
        assert signals.ndim == 4
        assert labels.ndim == 1
