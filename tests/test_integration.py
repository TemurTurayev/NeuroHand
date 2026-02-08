"""Integration test: end-to-end pipeline with synthetic data."""

import numpy as np
import pytest
import torch

from src.constants import N_CHANNELS, N_CLASSES, N_SAMPLES
from src.data.dataset import create_data_loaders
from src.models.eegnet import EEGNet
from src.models.utils import save_checkpoint, load_checkpoint
from src.training.config import TrainingConfig
from src.training.train import Trainer, set_seed
from src.training.evaluate import ModelEvaluator, calculate_itr
from src.inference.predict import EEGPredictor


@pytest.mark.integration
class TestEndToEnd:
    """Full pipeline: create data -> train -> evaluate -> predict."""

    def test_full_pipeline(self, tmp_data_dir):
        set_seed(42)

        config = TrainingConfig(
            epochs=3,
            batch_size=8,
            device="cpu",
            verbose=False,
            save_dir=str(tmp_data_dir / "checkpoints"),
            early_stopping_patience=10,
        )

        train_loader, test_loader = create_data_loaders(
            data_path=str(tmp_data_dir), batch_size=8
        )

        model = EEGNet(verbose=False)
        trainer = Trainer(model, train_loader, test_loader, config)

        history = trainer.train()
        assert len(history['train_loss']) == 3
        assert all(isinstance(v, float) for v in history['train_loss'])

        model_on_cpu = model.to("cpu")
        evaluator = ModelEvaluator(
            model=model_on_cpu, test_loader=test_loader, device='cpu'
        )
        results = evaluator.evaluate()

        assert 'accuracy' in results
        assert 'kappa' in results
        assert 'itr' in results
        assert results['confusion_matrix'].shape == (N_CLASSES, N_CLASSES)

        checkpoint_path = tmp_data_dir / "checkpoints" / "integration_test.pth"
        save_checkpoint(
            model=model_on_cpu,
            optimizer=trainer.optimizer,
            epoch=3,
            loss=history['train_loss'][-1],
            accuracy=results['accuracy'] * 100,
            filepath=str(checkpoint_path),
        )
        assert checkpoint_path.exists()

        predictor = EEGPredictor(model_path=str(checkpoint_path), device='cpu')
        signal = np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float32)
        result = predictor.predict(signal)

        assert 'class_id' in result
        assert 0 <= result['class_id'] < N_CLASSES
        assert 0 <= result['confidence'] <= 1.0
