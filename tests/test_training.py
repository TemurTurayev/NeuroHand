"""Tests for training infrastructure."""

import numpy as np
import pytest
import torch

from src.constants import N_CHANNELS, N_CLASSES, N_SAMPLES
from src.data.dataset import create_data_loaders
from src.models.eegnet import EEGNet
from src.training.config import TrainingConfig
from src.training.train import Trainer, set_seed


class TestSetSeed:
    """Tests for reproducibility seed management."""

    def test_torch_reproducible(self):
        set_seed(123)
        a = torch.randn(10)
        set_seed(123)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_numpy_reproducible(self):
        set_seed(123)
        a = np.random.randn(10)
        set_seed(123)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.n_classes == N_CLASSES
        assert config.n_channels == N_CHANNELS
        assert config.epochs == 300
        assert config.batch_size == 64

    def test_custom_values(self):
        config = TrainingConfig(epochs=10, batch_size=16, learning_rate=0.01)
        assert config.epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 0.01

    def test_save_dir_created(self, tmp_path):
        config = TrainingConfig(save_dir=str(tmp_path / "checkpoints"))
        assert config.save_dir.exists()


class TestTrainer:
    """Tests for the Trainer class."""

    @pytest.fixture
    def trainer(self, tmp_data_dir):
        config = TrainingConfig(
            epochs=2,
            batch_size=8,
            device="cpu",
            verbose=False,
            save_dir=str(tmp_data_dir / "checkpoints"),
        )
        train_loader, test_loader = create_data_loaders(
            data_path=str(tmp_data_dir), batch_size=8
        )
        model = EEGNet(verbose=False)
        return Trainer(model, train_loader, test_loader, config)

    def test_train_epoch_returns_metrics(self, trainer):
        loss, acc = trainer.train_epoch(0)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100

    def test_evaluate_returns_metrics(self, trainer):
        loss, acc = trainer.evaluate(0)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100

    def test_full_train_returns_history(self, trainer):
        history = trainer.train()
        assert 'train_loss' in history
        assert 'test_acc' in history
        assert len(history['train_loss']) == 2
