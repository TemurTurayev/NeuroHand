"""
NeuroHand Constants
====================

Central location for all project-wide constants.
Eliminates hardcoded values scattered across the codebase.
"""

from typing import Dict, Tuple

# Motor imagery class definitions
CLASS_NAMES: Tuple[str, ...] = ('Left Hand', 'Right Hand', 'Feet', 'Tongue')
N_CLASSES: int = 4
CLASS_MAPPING: Dict[str, int] = {
    'left_hand': 0,
    'right_hand': 1,
    'feet': 2,
    'tongue': 3,
}

# EEG signal parameters (BCI Competition IV-2a defaults)
N_CHANNELS: int = 22
N_SAMPLES: int = 1000
SAMPLING_RATE: int = 250
EPOCH_DURATION: float = 4.0

# Frequency bands for bandpass filter (Hz)
LOWCUT: float = 4.0
HIGHCUT: float = 38.0

# EEG frequency band definitions (for visualization and analysis)
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Low Gamma': (30.0, 38.0),
}

# Number of subjects in BCI Competition IV-2a
N_SUBJECTS: int = 9
