"""
BCI Lab Export
==============

Готовит данные для браузерной лаборатории `bci-lab/index.html`:

1. Загружает BCI Competition IV 2a (сессии T, 9 испытуемых) из .npz-зеркала
   (https://github.com/bregydoc/bcidatasetIV2a — порт оригинальных GDF-файлов).
2. Фильтрует 4–38 Hz и нарезает эпохи 0–4 s после подсказки (как в
   `src/data/preprocessing.py`), стандартизует каждый канал каждой попытки.
3. Для каждого испытуемого обучает EEGNet из `src/models/eegnet.py` на восьми
   других («новый пользователь», leave-one-subject-out), затем калибрует её
   на обучающих попытках самого испытуемого (transfer learning).
   80 % попыток каждого испытуемого — обучение/калибровка, 20 % — тест.
4. Считает ERD/ERS (десинхронизацию) в мю- (8–13 Hz) и бета- (13–30 Hz)
   диапазонах: среднее по классам и для каждой демонстрационной попытки.
5. Пишет всё в `bci-lab/lab-data.js` (`window.NEUROHAND_LAB = {...}`), чтобы
   страница открывалась даже с file:// без сервера.

Данные открытые и обезличенные (испытуемые A01–A09), персональной
медицинской информации в экспорте нет.

Usage:
    python -m src.visualization.bci_lab_export --data-dir data/raw/bciiv2a_npz
    python -m src.visualization.bci_lab_export --reuse-checkpoints  # skip base training

Автор: Temur Turayev
TashPMI, 2026
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal

from src.constants import CLASS_NAMES, HIGHCUT, LOWCUT, N_CHANNELS, SAMPLING_RATE
from src.logging_config import get_logger, setup_logging
from src.models.eegnet import EEGNet

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# BCI IV 2a channel order (first 22 columns of the signal; 3 EOG follow)
CHANNEL_NAMES: Tuple[str, ...] = (
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz',
)

# Schematic 2D scalp positions in 10 % steps of the 10-20 system:
# Cz at the centre, the head outline (T7/T8 level) at radius 4, nose at +y.
CHANNEL_POSITIONS: Tuple[Tuple[int, int], ...] = (
    (0, 2),
    (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
    (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
    (-1, -2), (0, -2), (1, -2),
    (0, -3),
)

# GDF event codes for the four motor imagery cues
CUE_CODES: Dict[int, int] = {769: 0, 770: 1, 771: 2, 772: 3}

# Time windows, in seconds relative to the cue.
# Trial paradigm: fixation cross at -2 s, cue at 0 s, imagery until +4 s.
MODEL_WINDOW = (0.0, 4.0)       # what EEGNet sees (1000 samples)
DISPLAY_WINDOW = (-1.5, 4.0)    # what the lab plays back (1375 samples)
BASELINE_WINDOW = (-1.5, -0.5)  # reference period for ERD %

MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
ERD_FRAME_STEP = 10             # 250 Hz / 10 = 25 ERD frames per second
ERD_SMOOTHING_S = 0.5           # moving-average window for band power


def _samples(seconds: float, fs: int = SAMPLING_RATE) -> int:
    return int(round(seconds * fs))


def load_subject(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one BCI IV 2a training session from the .npz port.

    Args:
        npz_path: Path to AxxT.npz

    Returns:
        eeg: Continuous EEG in µV [22, n_samples]
        cues: Cue onset sample indices (0-based) [n_trials]
        labels: Class labels 0..3 [n_trials] (expert-rejected trials removed)
    """
    data = np.load(npz_path)
    eeg = np.nan_to_num(data['s'][:, :N_CHANNELS].T.astype(np.float64))

    event_types = data['etyp'].ravel()
    # GDF positions are 1-based
    event_positions = data['epos'].ravel().astype(np.int64) - 1

    cue_mask = np.isin(event_types, list(CUE_CODES))
    cues = event_positions[cue_mask]
    labels = np.array([CUE_CODES[int(code)] for code in event_types[cue_mask]])

    # 'artifacts' flags trials that the dataset authors marked as contaminated
    if 'artifacts' in data.files and len(data['artifacts'].ravel()) == len(cues):
        keep = data['artifacts'].ravel() == 0
        cues, labels = cues[keep], labels[keep]

    return eeg, cues, labels


def bandpass(eeg: np.ndarray, low: float, high: float, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth band-pass on continuous EEG [channels, samples]."""
    sos = scipy_signal.butter(order, [low, high], btype='band', fs=SAMPLING_RATE,
                              output='sos')
    return scipy_signal.sosfiltfilt(sos, eeg, axis=-1)


def epoch(continuous: np.ndarray, cues: np.ndarray,
          window: Tuple[float, float]) -> np.ndarray:
    """Cut [trials, channels, samples] around each cue."""
    start, stop = _samples(window[0]), _samples(window[1])
    return np.stack([continuous[:, cue + start:cue + stop] for cue in cues])


def standardize(trials: np.ndarray) -> np.ndarray:
    """Per-trial, per-channel z-score (same as EEGPreprocessor.normalize)."""
    mean = trials.mean(axis=-1, keepdims=True)
    std = trials.std(axis=-1, keepdims=True)
    std[std == 0] = 1
    return (trials - mean) / std


def band_power_envelope(eeg: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """
    Instantaneous band power of continuous EEG, smoothed over ERD_SMOOTHING_S.

    Medical context:
        Motor imagery suppresses the sensorimotor mu and beta rhythms over the
        contralateral hand area (event-related desynchronization, ERD).
        Squaring the band-passed signal gives its power; smoothing turns it
        into an envelope we can compare with the resting baseline.
    """
    filtered = bandpass(eeg, band[0], band[1], order=4)
    power = filtered ** 2
    kernel = np.ones(_samples(ERD_SMOOTHING_S)) / _samples(ERD_SMOOTHING_S)
    return scipy_signal.fftconvolve(power, kernel[None, :], mode='same', axes=-1)


def erd_percent(power_epochs: np.ndarray) -> np.ndarray:
    """
    Convert band power epochs to ERD/ERS % relative to the baseline.

    ERD% = (P(t) - R) / R * 100, where R is the mean power in BASELINE_WINDOW
    (Pfurtscheller & Lopes da Silva, 1999). Negative = desynchronization.

    Args:
        power_epochs: [..., channels, samples] cut with DISPLAY_WINDOW

    Returns:
        ERD % decimated to ERD_FRAME_STEP [..., channels, frames]
    """
    offset = _samples(DISPLAY_WINDOW[0])
    b0 = _samples(BASELINE_WINDOW[0]) - offset
    b1 = _samples(BASELINE_WINDOW[1]) - offset
    reference = power_epochs[..., b0:b1].mean(axis=-1, keepdims=True)
    reference = np.maximum(reference, 1e-12)
    erd = (power_epochs - reference) / reference * 100.0
    return erd[..., ::ERD_FRAME_STEP]


def load_dataset(data_dir: Path, seed: int = 42) -> Dict[str, list]:
    """
    Load all subjects, preprocess and split 80/20 per subject (stratified).

    Returns:
        Dict with model-ready epochs, display epochs, band powers and split masks
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, list] = {
        'model': [], 'display': [], 'mu': [], 'beta': [],
        'labels': [], 'subjects': [], 'is_test': [],
    }

    for subject in range(1, 10):
        path = data_dir / f"A{subject:02d}T.npz"
        if not path.exists():
            logger.warning("Missing %s, skipping subject %d", path, subject)
            continue

        eeg, cues, labels = load_subject(path)
        filtered = bandpass(eeg, LOWCUT, HIGHCUT)
        mu_power = band_power_envelope(eeg, MU_BAND)
        beta_power = band_power_envelope(eeg, BETA_BAND)

        # Test set: 20 % of each class, chosen per subject
        is_test = np.zeros(len(labels), dtype=bool)
        for cls in range(len(CLASS_NAMES)):
            idx = np.flatnonzero(labels == cls)
            n_test = int(round(len(idx) * 0.2))
            is_test[rng.choice(idx, size=n_test, replace=False)] = True

        # float32 keeps ~2600 trials x 4 arrays within a few GB of RAM
        out['model'].append(
            standardize(epoch(filtered, cues, MODEL_WINDOW)).astype(np.float32))
        out['display'].append(epoch(filtered, cues, DISPLAY_WINDOW).astype(np.float32))
        out['mu'].append(epoch(mu_power, cues, DISPLAY_WINDOW).astype(np.float32))
        out['beta'].append(epoch(beta_power, cues, DISPLAY_WINDOW).astype(np.float32))
        out['labels'].append(labels)
        out['subjects'].append(np.full(len(labels), subject))
        out['is_test'].append(is_test)

        logger.info("Subject A%02d: %d trials (%d test)", subject, len(labels),
                    is_test.sum())

    return {key: np.concatenate(value) for key, value in out.items()}


def train_eegnet(x_train: np.ndarray, y_train: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 300, patience: int = 60, seed: int = 42,
                 init_state: Optional[Dict[str, torch.Tensor]] = None,
                 lr: float = 1e-3, batch_size: int = 64,
                 log_every: int = 10) -> Tuple[EEGNet, int]:
    """
    Train EEGNet on CPU with Adam, max-norm and early stopping.

    The best epoch is chosen by validation accuracy, ties broken by lower
    validation loss (small validation sets produce many ties).

    Args:
        init_state: Start from these weights (used for per-subject calibration)

    Returns:
        Trained model in eval mode and the best epoch index
    """
    torch.manual_seed(seed)
    model = EEGNet(n_classes=len(CLASS_NAMES), n_channels=N_CHANNELS,
                   n_samples=x_train.shape[-1])
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    xt = torch.from_numpy(x_train[:, None].astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.int64))
    xv = torch.from_numpy(x_val[:, None].astype(np.float32))
    yv = torch.from_numpy(y_val.astype(np.int64))

    best_key, best_state, best_epoch, stale = None, None, 0, 0
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            model.apply_max_norm_constraint()

        model.eval()
        with torch.no_grad():
            logits = model(xv)
            val_acc = (logits.argmax(1) == yv).float().mean().item()
            val_loss = loss_fn(logits, yv).item()
        key = (val_acc, -val_loss)
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if log_every and ep % log_every == 0:
            logger.info("epoch %3d  loss %.3f  val_acc %.3f  best %.3f (epoch %d)",
                        ep, loss.item(), val_acc, best_key[0], best_epoch)
        if stale >= patience:
            logger.info("Early stopping at epoch %d (best %d)", ep, best_epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def split_validation(indices: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle indices and split off a validation part: returns (fit, val)."""
    indices = indices.copy()
    np.random.default_rng(seed).shuffle(indices)
    n_val = max(int(len(indices) * fraction), 1)
    return indices[n_val:], indices[:n_val]


def fine_tune(model: EEGNet, x: np.ndarray, y: np.ndarray, epochs: int,
              lr: float = 5e-4, batch_size: int = 32, seed: int = 0,
              x_eval: Optional[np.ndarray] = None,
              y_eval: Optional[np.ndarray] = None) -> List[float]:
    """
    Continue training `model` in place for a fixed number of epochs.

    Returns:
        Accuracy on (x_eval, y_eval) before training and after every epoch
        (empty list when no evaluation set is given)
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x[:, None].astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    curve: List[float] = []

    def evaluate_now() -> None:
        if x_eval is not None:
            model.eval()
            curve.append(float((predict_proba(model, x_eval).argmax(1) == y_eval).mean()))

    evaluate_now()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(xt[batch]), yt[batch]).backward()
            optimizer.step()
            model.apply_max_norm_constraint()
        evaluate_now()
    model.eval()
    return curve


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    """Split positions 0..len(labels)-1 into class-balanced folds."""
    rng = np.random.default_rng(seed)
    folds: List[list] = [[] for _ in range(n_folds)]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % n_folds].append(int(i))
    return [np.array(sorted(f)) for f in folds]


def train_new_user_models(data: Dict[str, np.ndarray], checkpoint_dir: Path,
                          epochs: int = 120, patience: int = 25,
                          reuse: bool = False) -> Dict[int, EEGNet]:
    """
    Leave-one-subject-out base models: for every subject, an EEGNet trained only
    on the other eight people. This is what a brand-new NeuroHand user gets
    before any calibration.
    """
    models: Dict[int, EEGNet] = {}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for subject in np.unique(data['subjects']):
        path = checkpoint_dir / f"bci_lab_base_A{int(subject):02d}.pth"
        if reuse and path.exists():
            model = EEGNet(n_classes=len(CLASS_NAMES), n_channels=N_CHANNELS,
                           n_samples=data['model'].shape[-1])
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            logger.info("A%02d: loaded base model from %s", subject, path)
        else:
            others = np.flatnonzero(~data['is_test'] & (data['subjects'] != subject))
            fit, val = split_validation(others, 0.1, seed=int(subject))
            model, best_epoch = train_eegnet(
                data['model'][fit], data['labels'][fit],
                data['model'][val], data['labels'][val],
                epochs=epochs, patience=patience, seed=int(subject), log_every=20)
            torch.save(model.state_dict(), path)
            logger.info("A%02d: base model trained on %d trials of 8 other subjects "
                        "(best epoch %d)", subject, len(fit), best_epoch)
        models[int(subject)] = model
    return models


def calibrate_subjects(data: Dict[str, np.ndarray], base_models: Dict[int, EEGNet],
                       max_epochs: int = 40, n_folds: int = 3) -> Tuple[Dict[int, EEGNet], Dict[int, int]]:
    """
    Fine-tune each subject's base model on that subject's training trials.

    This is the transfer-learning step NeuroHand plans for OpenBCI users:
    start from a model that already knows motor imagery in general (from
    other people), then adapt it to one brain with a calibration session.

    How long to fine-tune is chosen by n-fold cross-validation inside the
    subject's training trials (0 epochs = keep the base model). The base
    model never saw this subject, so the folds are honest, and the held-out
    test trials never influence the choice.

    Returns:
        Calibrated model per subject and the number of epochs chosen
    """
    models: Dict[int, EEGNet] = {}
    chosen: Dict[int, int] = {}
    for subject in np.unique(data['subjects']):
        base_state = base_models[int(subject)].state_dict()

        def fresh() -> EEGNet:
            model = EEGNet(n_classes=len(CLASS_NAMES), n_channels=N_CHANNELS,
                           n_samples=data['model'].shape[-1])
            model.load_state_dict(base_state)
            return model

        own = np.flatnonzero(~data['is_test'] & (data['subjects'] == subject))
        x, y = data['model'][own], data['labels'][own]

        curves = []
        for k, fold in enumerate(stratified_folds(y, n_folds, seed=int(subject))):
            fit = np.setdiff1d(np.arange(len(y)), fold)
            curves.append(fine_tune(fresh(), x[fit], y[fit], max_epochs, seed=k,
                                    x_eval=x[fold], y_eval=y[fold]))
        mean_curve = np.mean(curves, axis=0)
        # Light smoothing: single-epoch spikes on ~55 trials are mostly noise
        smooth = np.convolve(np.pad(mean_curve, 2, mode='edge'), np.ones(5) / 5, mode='valid')
        best_epochs = int(np.argmax(smooth))

        model = fresh()
        fine_tune(model, x, y, best_epochs, seed=int(subject))
        models[int(subject)] = model
        chosen[int(subject)] = best_epochs
        logger.info("A%02d: CV accuracy %.3f -> %.3f after %d epochs of calibration",
                    subject, mean_curve[0], mean_curve[best_epochs], best_epochs)
    return models, chosen


def predict_proba(model: EEGNet, x: np.ndarray) -> np.ndarray:
    """Softmax probabilities for standardized epochs [n, 22, 1000]."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x[:, None].astype(np.float32)))
        return F.softmax(logits, dim=1).numpy()


def encode_int16(values: np.ndarray) -> Tuple[str, float]:
    """Quantize to int16 with a per-array scale; return base64 + scale."""
    scale = float(np.abs(values).max() / 32000.0) or 1.0
    q = np.round(values / scale).astype('<i2')
    return base64.b64encode(q.tobytes()).decode('ascii'), scale


def decode_int16(b64: str, scale: float, shape: Tuple[int, ...]) -> np.ndarray:
    """Inverse of encode_int16 (used to mirror what the browser sees)."""
    q = np.frombuffer(base64.b64decode(b64), dtype='<i2').astype(np.float64)
    return (q * scale).reshape(shape)


def export_weights(model: EEGNet) -> Dict[str, list]:
    """Flatten every tensor the browser needs, rounded to 6 significant digits."""
    state = model.state_dict()
    keep = [k for k in state if not k.endswith('num_batches_tracked')]
    return {k: [float(f"{v:.6g}") for v in state[k].flatten().tolist()] for k in keep}


def pick_demo_trials(data: Dict[str, np.ndarray], per_class: int,
                     seed: int = 7) -> List[int]:
    """Random test trials, per_class of each class, spread over subjects."""
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for cls in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(data['is_test'] & (data['labels'] == cls))
        rng.shuffle(idx)
        used_subjects: set = set()
        # First pass: one trial per subject, then fill up
        for i in idx:
            if len([c for c in chosen if data['labels'][c] == cls]) >= per_class:
                break
            if data['subjects'][i] not in used_subjects:
                chosen.append(int(i))
                used_subjects.add(data['subjects'][i])
    return chosen


def evaluate(pred: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> dict:
    """Accuracy, Cohen's kappa inputs, confusion matrix and per-subject accuracy."""
    n_cls = len(CLASS_NAMES)
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    for true, guess in zip(y, pred):
        confusion[true, guess] += 1
    per_subject = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        per_subject.append({
            'subject': f"A{int(subject):02d}",
            'accuracy': round(float((pred[mask] == y[mask]).mean()), 4),
            'n': int(mask.sum()),
        })
    return {
        'accuracy': round(float((pred == y).mean()), 4),
        'confusion': confusion.tolist(),
        'per_subject': per_subject,
    }


def grand_average_erd(data: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, list]:
    """
    ERD % per band and class, averaged over subjects with equal weight.

    Within a subject: average band power over trials, then express it relative
    to the baseline (classic ERD method). Across subjects: mean of ERD %, so a
    subject with a strong mu rhythm does not dominate the atlas.
    """
    grand: Dict[str, list] = {}
    for band in ('mu', 'beta'):
        grand[band] = []
        for cls in range(len(CLASS_NAMES)):
            per_subject = []
            for subject in np.unique(data['subjects']):
                sel = mask & (data['labels'] == cls) & (data['subjects'] == subject)
                if sel.any():
                    per_subject.append(erd_percent(data[band][sel].mean(axis=0)))
            grand[band].append(np.round(np.mean(per_subject, axis=0), 1).tolist())
    return grand


def build_export(data: Dict[str, np.ndarray], base_models: Dict[int, EEGNet],
                 calibrated: Dict[int, EEGNet], per_class: int) -> dict:
    """Assemble the JSON payload for the browser lab."""
    test = data['is_test']
    x_test, y_test, subj_test = data['model'][test], data['labels'][test], data['subjects'][test]

    pred_new_user = np.empty(len(y_test), dtype=int)
    pred_calibrated = np.empty(len(y_test), dtype=int)
    for subject in calibrated:
        mask = subj_test == subject
        pred_new_user[mask] = predict_proba(base_models[subject], x_test[mask]).argmax(1)
        pred_calibrated[mask] = predict_proba(calibrated[subject], x_test[mask]).argmax(1)

    display_len = data['display'].shape[-1]
    trials = []
    for i in pick_demo_trials(data, per_class):
        b64, scale = encode_int16(data['display'][i])
        # Mirror the browser exactly: dequantize -> cut model window -> z-score
        seen = decode_int16(b64, scale, (N_CHANNELS, display_len))
        cue = -_samples(DISPLAY_WINDOW[0])
        x = standardize(seen[None, :, cue:cue + _samples(MODEL_WINDOW[1])])
        subject = int(data['subjects'][i])

        erd = {}
        for band in ('mu', 'beta'):
            values = np.clip(erd_percent(data[band][i]), -100, 127)
            erd[band] = base64.b64encode(
                np.round(values).astype(np.int8).tobytes()).decode('ascii')

        trials.append({
            'subject': f"A{subject:02d}",
            'label': int(data['labels'][i]),
            'signal': b64,
            'scale': scale,
            'erd': erd,
            'reference_proba': {
                'new_user': [round(float(p), 6) for p in predict_proba(base_models[subject], x)[0]],
                'calibrated': [round(float(p), 6) for p in predict_proba(calibrated[subject], x)[0]],
            },
        })

    n_frames = len(range(0, display_len, ERD_FRAME_STEP))
    any_model = next(iter(calibrated.values()))
    return {
        'meta': {
            'dataset': 'BCI Competition IV 2a (Graz), training sessions A01T–A09T',
            'fs': SAMPLING_RATE,
            'channels': list(CHANNEL_NAMES),
            'positions': [list(p) for p in CHANNEL_POSITIONS],
            'classes': list(CLASS_NAMES),
            'bandpass': [LOWCUT, HIGHCUT],
            'display_window': list(DISPLAY_WINDOW),
            'model_window': list(MODEL_WINDOW),
            'baseline_window': list(BASELINE_WINDOW),
            'display_samples': display_len,
            'erd_frames': n_frames,
            'erd_frame_step': ERD_FRAME_STEP,
            'bands': {'mu': list(MU_BAND), 'beta': list(BETA_BAND)},
            'generated': time.strftime('%Y-%m-%d'),
        },
        'model': {
            'architecture': {'F1': any_model.F1, 'D': any_model.D, 'F2': any_model.F2,
                             'kernel_length': any_model.kernel_length,
                             'n_samples': any_model.n_samples},
            'n_params': any_model.count_parameters(),
            'subject_weights': {
                f"A{s:02d}": {'new_user': export_weights(base_models[s]),
                              'calibrated': export_weights(calibrated[s])}
                for s in calibrated},
        },
        'evaluation': {
            'n_test': int(test.sum()),
            'n_train': int((~test).sum()),
            'calibration_trials_per_subject': round(float((~test).sum()) / len(calibrated), 1),
            'chance': round(1 / len(CLASS_NAMES), 4),
            'new_user': evaluate(pred_new_user, y_test, subj_test),
            'calibrated': evaluate(pred_calibrated, y_test, subj_test),
        },
        'grand_average_erd': grand_average_erd(data, ~test),
        'trials': trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--data-dir', type=Path,
                        default=PROJECT_ROOT / 'data' / 'raw' / 'bciiv2a_npz',
                        help='Folder with A01T.npz … A09T.npz')
    parser.add_argument('--out', type=Path,
                        default=PROJECT_ROOT / 'bci-lab' / 'lab-data.js')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Max epochs for each leave-one-subject-out base model')
    parser.add_argument('--calibration-epochs', type=int, default=40,
                        help='Upper bound; the actual number is chosen by CV')
    parser.add_argument('--per-class', type=int, default=6,
                        help='Demo trials per class shipped to the browser')
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=PROJECT_ROOT / 'models' / 'checkpoints')
    parser.add_argument('--reuse-checkpoints', action='store_true',
                        help='Load base models from --checkpoint-dir when present')
    args = parser.parse_args()
    setup_logging(name=logger.name)

    data = load_dataset(args.data_dir)
    base_models = train_new_user_models(data, args.checkpoint_dir, epochs=args.epochs,
                                        reuse=args.reuse_checkpoints)
    calibrated, calibration_epochs = calibrate_subjects(
        data, base_models, max_epochs=args.calibration_epochs)

    payload = build_export(data, base_models, calibrated, args.per_class)
    payload['evaluation']['calibration_epochs'] = {
        f"A{s:02d}": e for s, e in calibration_epochs.items()}
    ev = payload['evaluation']
    logger.info("Test accuracy on %d trials: new user %.3f, calibrated %.3f",
                ev['n_test'], ev['new_user']['accuracy'], ev['calibrated']['accuracy'])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        '// Generated by src/visualization/bci_lab_export.py — do not edit by hand.\n'
        'window.NEUROHAND_LAB = ' + json.dumps(payload, separators=(',', ':')) + ';\n',
        encoding='utf-8')
    logger.info("Wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)


if __name__ == '__main__':
    main()
