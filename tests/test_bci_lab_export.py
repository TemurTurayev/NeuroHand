"""
Tests for the browser BCI lab export (src/visualization/bci_lab_export.py)
and for the JavaScript EEGNet port (bci-lab/eegnet.js).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.eegnet import EEGNet
from src.visualization.bci_lab_export import (
    BASELINE_WINDOW,
    CHANNEL_NAMES,
    CHANNEL_POSITIONS,
    DISPLAY_WINDOW,
    ERD_FRAME_STEP,
    decode_int16,
    encode_int16,
    erd_percent,
    export_weights,
    load_subject,
    standardize,
    stratified_folds,
)

PROJECT_ROOT = Path(__file__).parent.parent
EEGNET_JS = PROJECT_ROOT / "bci-lab" / "eegnet.js"


def _write_fake_session(path: Path, n_samples: int = 20000) -> None:
    """Minimal AxxT.npz with the same keys and event layout as the real port."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((n_samples, 25))
    # trial start (768) followed by its cue, GDF positions are 1-based
    etyp = np.array([32766, 768, 769, 768, 772, 768, 770])[:, None]
    epos = np.array([1, 1001, 1501, 5001, 5501, 9001, 9501])[:, None]
    edur = np.array([0, 1875, 313, 1875, 313, 1875, 313])[:, None]
    artifacts = np.array([0, 1, 0])[:, None]  # second trial rejected
    np.savez(path, s=signal, etyp=etyp, epos=epos, edur=edur, artifacts=artifacts)


class TestLoadSubject:
    def test_cues_are_zero_based_and_rejected_trials_removed(self, tmp_path):
        path = tmp_path / "A01T.npz"
        _write_fake_session(path)

        eeg, cues, labels = load_subject(path)

        assert eeg.shape == (22, 20000)
        np.testing.assert_array_equal(cues, [1500, 9500])
        np.testing.assert_array_equal(labels, [0, 1])  # left hand, right hand


class TestSignalHelpers:
    def test_channel_montage_is_consistent(self):
        assert len(CHANNEL_NAMES) == len(CHANNEL_POSITIONS) == 22
        assert CHANNEL_POSITIONS[CHANNEL_NAMES.index('Cz')] == (0, 0)
        # C3 over the left hemisphere, C4 over the right
        assert CHANNEL_POSITIONS[CHANNEL_NAMES.index('C3')][0] < 0
        assert CHANNEL_POSITIONS[CHANNEL_NAMES.index('C4')][0] > 0

    def test_standardize_zero_mean_unit_variance(self, rng):
        trials = rng.standard_normal((3, 22, 1000)) * 20 + 5
        z = standardize(trials)
        np.testing.assert_allclose(z.mean(axis=-1), 0, atol=1e-9)
        np.testing.assert_allclose(z.std(axis=-1), 1, atol=1e-9)

    def test_erd_percent_detects_power_drop(self):
        fs = 250
        n = int((DISPLAY_WINDOW[1] - DISPLAY_WINDOW[0]) * fs)
        power = np.ones((1, n))
        cue = int(-DISPLAY_WINDOW[0] * fs)
        power[:, cue:] = 0.5  # power halves after the cue -> ERD of -50 %

        erd = erd_percent(power)

        assert erd.shape == (1, len(range(0, n, ERD_FRAME_STEP)))
        baseline_frame = int((BASELINE_WINDOW[0] - DISPLAY_WINDOW[0]) * fs) // ERD_FRAME_STEP
        assert erd[0, baseline_frame] == pytest.approx(0.0)
        assert erd[0, -1] == pytest.approx(-50.0)

    def test_stratified_folds_cover_every_trial_once(self):
        labels = np.array([0, 1, 2, 3] * 10 + [0, 1])
        folds = stratified_folds(labels, n_folds=4, seed=1)

        together = np.sort(np.concatenate(folds))
        np.testing.assert_array_equal(together, np.arange(len(labels)))
        for fold in folds:
            # every class is represented in every fold
            assert set(labels[fold]) == {0, 1, 2, 3}

    def test_int16_round_trip(self, rng):
        values = rng.standard_normal((22, 1375)) * 15
        b64, scale = encode_int16(values)
        restored = decode_int16(b64, scale, values.shape)
        assert np.abs(restored - values).max() <= scale / 2 + 1e-12


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
class TestJavaScriptPort:
    def test_eegnet_js_matches_pytorch(self, tmp_path):
        torch.manual_seed(3)
        model = EEGNet(n_classes=4, n_channels=22, n_samples=1000)
        # Non-trivial batch-norm statistics so the test covers them too
        for bn in (model.batchnorm1, model.batchnorm2, model.batchnorm3):
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
            bn.weight.data.uniform_(0.5, 1.5)
            bn.bias.data.uniform_(-0.2, 0.2)
        model.fc.weight.data.normal_(0, 0.05)
        model.eval()

        rng = np.random.default_rng(11)
        x = standardize(rng.standard_normal((2, 22, 1000)))
        with torch.no_grad():
            expected = F.softmax(model(torch.from_numpy(x[:, None].astype(np.float32))), dim=1).numpy()

        payload = {
            'weights': export_weights(model),
            'arch': {'F1': 8, 'D': 2, 'F2': 16, 'kernel_length': 64, 'n_samples': 1000},
            'x': x.tolist(),
        }
        (tmp_path / "input.json").write_text(json.dumps(payload))
        script = (
            "const fs = require('fs');"
            f"const E = require({json.dumps(str(EEGNET_JS))});"
            f"const p = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'input.json'))}));"
            "const m = new E.Model(p.weights, p.arch);"
            "const out = p.x.map(tr => m.forward(tr.map(ch => Float64Array.from(ch))).proba);"
            "process.stdout.write(JSON.stringify(out));"
        )
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                                check=True, timeout=60)
        got = np.array(json.loads(result.stdout))

        # float32 PyTorch vs float64 JS, weights rounded to 6 significant digits
        np.testing.assert_allclose(got, expected, atol=1e-4)
