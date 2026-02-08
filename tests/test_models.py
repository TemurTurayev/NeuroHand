"""Tests for EEGNet model."""

import torch
import pytest

from src.constants import N_CHANNELS, N_CLASSES, N_SAMPLES
from src.models.eegnet import EEGNet, create_model


class TestEEGNet:
    """Tests for the EEGNet architecture."""

    def test_forward_shape(self, model_input):
        model = EEGNet(verbose=False)
        output = model(model_input)
        assert output.shape == (model_input.shape[0], N_CLASSES)

    def test_forward_no_nan(self, model_input):
        model = EEGNet(verbose=False)
        output = model(model_input)
        assert not torch.isnan(output).any()

    def test_parameter_count(self):
        model = EEGNet(verbose=False)
        n_params = model.count_parameters()
        assert 1000 < n_params < 10000, f"Expected 1-10K params, got {n_params}"

    def test_max_norm_constraint(self):
        model = EEGNet(verbose=False)
        model.fc.weight.data.fill_(10.0)
        model.apply_max_norm_constraint()
        max_norm = model.fc.weight.norm(2, dim=1).max().item()
        assert max_norm <= model.norm_rate + 1e-6

    def test_different_channel_counts(self):
        for n_ch in [8, 16, 22, 32]:
            model = EEGNet(n_channels=n_ch, verbose=False)
            x = torch.randn(2, 1, n_ch, N_SAMPLES)
            output = model(x)
            assert output.shape == (2, N_CLASSES)

    def test_different_class_counts(self):
        for n_cls in [2, 3, 4, 6]:
            model = EEGNet(n_classes=n_cls, verbose=False)
            x = torch.randn(2, 1, N_CHANNELS, N_SAMPLES)
            output = model(x)
            assert output.shape == (2, n_cls)

    def test_mode_deterministic(self, model_input):
        model = EEGNet(verbose=False)
        model.eval()
        with torch.no_grad():
            out1 = model(model_input)
            out2 = model(model_input)
        assert torch.allclose(out1, out2)


class TestCreateModel:
    """Tests for the create_model helper."""

    def test_creates_on_cpu(self):
        model = create_model(device="cpu", verbose=False)
        assert isinstance(model, EEGNet)
        assert next(model.parameters()).device == torch.device("cpu")

    def test_respects_params(self):
        model = create_model(n_classes=2, n_channels=8, device="cpu", verbose=False)
        assert model.n_classes == 2
        assert model.n_channels == 8
