"""Tests for EEG preprocessing pipeline."""

import numpy as np
import pytest

from src.data.preprocessing import EEGPreprocessor


class TestBandpassFilter:
    """Tests for bandpass filtering."""

    def test_output_shape_preserved(self, sample_eeg):
        preprocessor = EEGPreprocessor(verbose=False)
        filtered = preprocessor.bandpass_filter(sample_eeg)
        assert filtered.shape == sample_eeg.shape

    def test_output_dtype_preserved(self, sample_eeg):
        preprocessor = EEGPreprocessor(verbose=False)
        filtered = preprocessor.bandpass_filter(sample_eeg)
        assert filtered.dtype == sample_eeg.dtype

    def test_removes_dc_offset(self, rng):
        preprocessor = EEGPreprocessor(verbose=False)
        signal = np.ones((22, 1000)) * 100 + rng.standard_normal((22, 1000)) * 0.1
        filtered = preprocessor.bandpass_filter(signal)
        assert abs(filtered.mean()) < 1.0


class TestNormalize:
    """Tests for normalization."""

    def test_standardize_zero_mean(self, sample_eeg):
        preprocessor = EEGPreprocessor(verbose=False)
        normalized = preprocessor.normalize(sample_eeg, method='standardize')
        channel_means = normalized.mean(axis=1)
        np.testing.assert_allclose(channel_means, 0.0, atol=1e-6)

    def test_standardize_unit_variance(self, sample_eeg):
        preprocessor = EEGPreprocessor(verbose=False)
        normalized = preprocessor.normalize(sample_eeg, method='standardize')
        channel_stds = normalized.std(axis=1)
        np.testing.assert_allclose(channel_stds, 1.0, atol=1e-6)

    def test_minmax_range(self, sample_eeg):
        preprocessor = EEGPreprocessor(verbose=False)
        normalized = preprocessor.normalize(sample_eeg, method='minmax')
        assert normalized.min() >= -1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_invalid_method_raises(self, sample_eeg):
        preprocessor = EEGPreprocessor(verbose=False)
        with pytest.raises(ValueError, match="Unknown normalization"):
            preprocessor.normalize(sample_eeg, method='invalid')

    def test_zero_std_channel(self):
        preprocessor = EEGPreprocessor(verbose=False)
        signal = np.zeros((22, 1000))
        signal[0, :] = 5.0
        normalized = preprocessor.normalize(signal, method='standardize')
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
