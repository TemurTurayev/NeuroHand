"""
EEG Signal Visualization
=========================

Функции для визуализации EEG сигналов.

Автор: Temur Turayev
TashPMI, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_eeg_channels(
    signal: np.ndarray,
    sampling_rate: int = 250,
    channel_names: Optional[List[str]] = None,
    title: str = "EEG Channels",
    figsize: tuple = (15, 10)
):
    """
    Plot multiple EEG channels.

    Args:
        signal: EEG data [n_channels, n_samples]
        sampling_rate: Sampling rate in Hz
        channel_names: List of channel names
        title: Plot title
        figsize: Figure size
    """
    n_channels, n_samples = signal.shape
    time = np.arange(n_samples) / sampling_rate

    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, signal[i], linewidth=0.5)
        ax.set_ylabel(channel_names[i], rotation=0, ha='right')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig


def plot_power_spectrum(
    signal: np.ndarray,
    sampling_rate: int = 250,
    channel_idx: int = 0,
    title: str = "Power Spectral Density",
    figsize: tuple = (12, 6)
):
    """
    Plot power spectral density of EEG signal.

    Args:
        signal: EEG data [n_channels, n_samples]
        sampling_rate: Sampling rate in Hz
        channel_idx: Channel index to plot
        title: Plot title
        figsize: Figure size
    """
    from scipy import signal as scipy_signal

    # Compute PSD
    freqs, psd = scipy_signal.welch(
        signal[channel_idx],
        fs=sampling_rate,
        nperseg=min(256, len(signal[channel_idx]))
    )

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(freqs, psd)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (V²/Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])

    # Mark frequency bands
    bands = {
        'Theta (4-8 Hz)': (4, 8, 'blue'),
        'Alpha (8-13 Hz)': (8, 13, 'green'),
        'Beta (13-30 Hz)': (13, 30, 'orange'),
        'Gamma (30-50 Hz)': (30, 50, 'red')
    }

    for name, (low, high, color) in bands.items():
        ax.axvspan(low, high, alpha=0.1, color=color, label=name)

    ax.legend()
    plt.tight_layout()

    return fig
