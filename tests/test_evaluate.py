"""Tests for model evaluation."""

import math

import numpy as np
import pytest

from src.training.evaluate import calculate_itr


class TestCalculateITR:
    """Tests for Information Transfer Rate calculation."""

    def test_perfect_accuracy(self):
        itr = calculate_itr(n_classes=4, accuracy=0.99)
        assert itr > 0

    def test_chance_level(self):
        itr = calculate_itr(n_classes=4, accuracy=0.25)
        assert abs(itr) < 1.0

    def test_zero_accuracy(self):
        itr = calculate_itr(n_classes=4, accuracy=0.0)
        assert itr == 0.0

    def test_full_accuracy(self):
        itr = calculate_itr(n_classes=4, accuracy=1.0)
        assert itr == 0.0

    def test_two_class(self):
        itr = calculate_itr(n_classes=2, accuracy=0.9)
        assert itr > 0

    def test_longer_trial_lower_itr(self):
        itr_short = calculate_itr(n_classes=4, accuracy=0.8, trial_duration=2.0)
        itr_long = calculate_itr(n_classes=4, accuracy=0.8, trial_duration=8.0)
        assert itr_short > itr_long
