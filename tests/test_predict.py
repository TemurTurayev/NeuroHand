"""Tests for inference/prediction pipeline."""

import numpy as np
import pytest
import torch

from src.constants import N_CHANNELS, N_CLASSES, N_SAMPLES
from src.inference.predict import EEGPredictor


class TestEEGPredictorPreprocess:
    """Tests for the preprocessing step of EEGPredictor."""

    @pytest.fixture
    def predictor_with_mock(self, tmp_path):
        """Create a predictor with a freshly saved dummy model."""
        from src.models.eegnet import EEGNet

        model = EEGNet(verbose=False)
        checkpoint = {
            'epoch': 1,
            'model_state_dict': model.state_dict(),
        }
        path = tmp_path / "test_model.pth"
        torch.save(checkpoint, path)
        return EEGPredictor(model_path=str(path), device='cpu')

    def test_preprocess_2d(self, predictor_with_mock, sample_eeg):
        tensor = predictor_with_mock.preprocess(sample_eeg)
        assert tensor.shape == (1, 1, N_CHANNELS, N_SAMPLES)

    def test_preprocess_3d(self, predictor_with_mock, batch_eeg):
        tensor = predictor_with_mock.preprocess(batch_eeg)
        assert tensor.shape == (8, 1, N_CHANNELS, N_SAMPLES)

    def test_predict_returns_dict(self, predictor_with_mock, sample_eeg):
        result = predictor_with_mock.predict(sample_eeg)
        assert 'class_id' in result
        assert 'class_name' in result
        assert 'confidence' in result
        assert 'inference_time_ms' in result
        assert 0 <= result['class_id'] < N_CLASSES
        assert 0 <= result['confidence'] <= 1.0

    def test_predict_batch(self, predictor_with_mock, batch_eeg):
        results = predictor_with_mock.predict_batch(batch_eeg)
        assert len(results) == 8
        for r in results:
            assert 'class_id' in r
            assert 'class_name' in r

    def test_probabilities_sum_to_one(self, predictor_with_mock, sample_eeg):
        result = predictor_with_mock.predict(sample_eeg, return_probs=True)
        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-4
