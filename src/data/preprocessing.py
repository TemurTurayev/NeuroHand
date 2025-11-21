"""
EEG Signal Preprocessing Pipeline
===================================

Обработка сырых EEG сигналов для обучения EEGNet:
- Bandpass фильтрация (4-38 Hz)
- Epoching (разбиение на эпохи)
- Artifact removal (удаление артефактов)
- Normalization (нормализация)
- Channel selection (выбор каналов)

Автор: Temur Turayev
TashPMI, 2024
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class EEGPreprocessor:
    """
    Preprocessor для EEG сигналов motor imagery.

    Этот класс обрабатывает сырые EEG данные и готовит их для обучения
    нейронной сети EEGNet.
    """

    def __init__(
        self,
        sampling_rate: int = 250,
        lowcut: float = 4.0,
        highcut: float = 38.0,
        n_channels: int = 22,
        epoch_duration: float = 4.0,
        verbose: bool = True
    ):
        """
        Initialize preprocessor.

        Args:
            sampling_rate: Частота дискретизации (Hz). Default: 250 Hz
            lowcut: Нижняя граница bandpass фильтра (Hz). Default: 4 Hz
            highcut: Верхняя граница bandpass фильтра (Hz). Default: 38 Hz
            n_channels: Количество EEG каналов. Default: 22
            epoch_duration: Длительность эпохи (seconds). Default: 4.0 s
            verbose: Печатать прогресс

        Frequency Bands Explanation:
            4-8 Hz:   Theta (motor preparation)
            8-13 Hz:  Alpha (motor imagery)
            13-30 Hz: Beta (active motor control)
            30-38 Hz: Low Gamma (motor execution)
        """
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.n_channels = n_channels
        self.epoch_duration = epoch_duration
        self.n_samples = int(sampling_rate * epoch_duration)  # 1000 samples at 250 Hz
        self.verbose = verbose

        # Class mapping (consistent with BCI Competition IV-2a)
        self.class_mapping = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'tongue': 3
        }

        if self.verbose:
            print(f"🔧 EEG Preprocessor initialized:")
            print(f"   Sampling rate: {sampling_rate} Hz")
            print(f"   Bandpass: {lowcut}-{highcut} Hz")
            print(f"   Channels: {n_channels}")
            print(f"   Epoch duration: {epoch_duration}s ({self.n_samples} samples)")

    def bandpass_filter(
        self,
        data: np.ndarray,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.

        Args:
            data: EEG data [channels, samples]
            order: Filter order (default: 5)

        Returns:
            Filtered data [channels, samples]

        Medical Context:
            Bandpass filter removes:
            - Low freq (<4 Hz): Slow drifts, DC offset
            - High freq (>38 Hz): EMG artifacts, powerline noise (50/60 Hz)
        """
        # Nyquist frequency
        nyq = 0.5 * self.sampling_rate

        # Normalized frequencies
        low = self.lowcut / nyq
        high = self.highcut / nyq

        # Design Butterworth filter
        b, a = scipy_signal.butter(order, [low, high], btype='band')

        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = scipy_signal.filtfilt(b, a, data[ch])

        return filtered_data

    def normalize(
        self,
        data: np.ndarray,
        method: str = 'standardize'
    ) -> np.ndarray:
        """
        Normalize EEG data.

        Args:
            data: EEG data [channels, samples]
            method: 'standardize' (zero mean, unit variance) or 'minmax'

        Returns:
            Normalized data

        Why Normalize?
            - Different channels have different voltage ranges
            - Neural networks train better with normalized inputs
            - Standardization: (x - mean) / std
        """
        if method == 'standardize':
            # Per-channel standardization (most common for EEG)
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            normalized = (data - mean) / std

        elif method == 'minmax':
            # Min-max scaling to [0, 1]
            min_val = np.min(data, axis=1, keepdims=True)
            max_val = np.max(data, axis=1, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            normalized = (data - min_val) / range_val

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def process_subject(
        self,
        subject_id: int,
        apply_filter: bool = True,
        apply_normalization: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all data for one subject.

        Args:
            subject_id: Subject number (1-9)
            apply_filter: Apply bandpass filter
            apply_normalization: Apply normalization

        Returns:
            X: Epochs [n_trials, n_channels, n_samples]
            y: Labels [n_trials]

        Processing Steps:
            1. Load raw data from MOABB
            2. Extract epochs (motor imagery periods)
            3. Apply bandpass filter (4-38 Hz)
            4. Normalize per channel
            5. Return as numpy arrays
        """
        if self.verbose:
            print(f"\n📊 Processing Subject {subject_id:02d}...")

        # Load data using MOABB
        dataset = BNCI2014_001()
        paradigm = MotorImagery(
            n_classes=4,
            channels=None,  # Use all EEG channels
            events=['left_hand', 'right_hand', 'feet', 'tongue'],
            tmin=0.0,  # Start of epoch (relative to cue)
            tmax=self.epoch_duration,  # End of epoch
            resample=self.sampling_rate
        )

        # Get epoched data
        X, labels, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[subject_id],
            return_epochs=False  # Return numpy arrays
        )

        if self.verbose:
            print(f"   Raw shape: {X.shape}")
            print(f"   Labels shape: {labels.shape}")

        # Apply bandpass filter
        if apply_filter:
            if self.verbose:
                print(f"   Applying bandpass filter ({self.lowcut}-{self.highcut} Hz)...")

            X_filtered = np.zeros_like(X)
            for trial in range(X.shape[0]):
                X_filtered[trial] = self.bandpass_filter(X[trial])
            X = X_filtered

        # Apply normalization
        if apply_normalization:
            if self.verbose:
                print(f"   Applying normalization...")

            X_normalized = np.zeros_like(X)
            for trial in range(X.shape[0]):
                X_normalized[trial] = self.normalize(X[trial])
            X = X_normalized

        # Convert labels to integers (0, 1, 2, 3)
        # Labels come as strings from MOABB, need to map them
        y = np.array([self.class_mapping.get(str(label), self.class_mapping.get(label, 0))
                      for label in labels])

        if self.verbose:
            print(f"   Final shape: {X.shape}")
            print(f"   Class distribution: {np.bincount(y)}")
            print(f"   ✅ Subject {subject_id:02d} processed!")

        return X, y

    def process_all_subjects(
        self,
        subjects: Optional[List[int]] = None,
        save_dir: Optional[str] = None
    ) -> dict:
        """
        Process all subjects and save to disk.

        Args:
            subjects: List of subject IDs (1-9). If None, process all.
            save_dir: Directory to save processed data. Default: data/processed/

        Returns:
            Dictionary with dataset statistics

        Output Files:
            - subject_XX_data.npy: EEG epochs [n_trials, n_channels, n_samples]
            - subject_XX_labels.npy: Labels [n_trials]
            - dataset_info.pkl: Metadata
        """
        if subjects is None:
            subjects = list(range(1, 10))  # All 9 subjects

        if save_dir is None:
            save_dir = PROJECT_ROOT / "data" / "processed"
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print("\n" + "="*60)
            print("🔄 PREPROCESSING ALL SUBJECTS")
            print("="*60)

        all_data = {}
        total_trials = 0

        # Process each subject
        for subject_id in tqdm(subjects, desc="Processing subjects"):
            try:
                X, y = self.process_subject(subject_id)

                # Save to disk
                np.save(save_dir / f"subject_{subject_id:02d}_data.npy", X)
                np.save(save_dir / f"subject_{subject_id:02d}_labels.npy", y)

                all_data[subject_id] = {
                    'n_trials': len(y),
                    'shape': X.shape,
                    'class_distribution': np.bincount(y).tolist()
                }

                total_trials += len(y)

            except Exception as e:
                print(f"❌ Error processing subject {subject_id}: {e}")
                continue

        # Save dataset info
        info = {
            'n_subjects': len(all_data),
            'subjects': list(all_data.keys()),
            'total_trials': total_trials,
            'n_classes': 4,
            'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'epoch_duration': self.epoch_duration,
            'lowcut': self.lowcut,
            'highcut': self.highcut,
            'subject_data': all_data
        }

        with open(save_dir / "dataset_info.pkl", 'wb') as f:
            pickle.dump(info, f)

        if self.verbose:
            print("\n" + "="*60)
            print("✅ PREPROCESSING COMPLETE!")
            print("="*60)
            print(f"   Total subjects: {info['n_subjects']}")
            print(f"   Total trials: {info['total_trials']}")
            print(f"   Saved to: {save_dir}")
            print("="*60)

        return info


def create_train_test_split(
    data_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> None:
    """
    Create train/test split from processed data.

    Args:
        data_dir: Directory with processed data
        test_size: Fraction for test set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
        verbose: Print information

    Creates:
        - train_data.npy: Training epochs
        - train_labels.npy: Training labels
        - test_data.npy: Test epochs
        - test_labels.npy: Test labels

    Split Strategy:
        - Subject-independent: Use subjects 1-7 for train, 8-9 for test
        - OR trial-based: Split trials 80/20 within each subject
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    if verbose:
        print("\n🔀 Creating train/test split...")

    # Load all subject data
    all_X = []
    all_y = []

    for subject_file in sorted(data_dir.glob("subject_*_data.npy")):
        subject_id = int(subject_file.stem.split('_')[1])
        X = np.load(subject_file)
        y = np.load(data_dir / f"subject_{subject_id:02d}_labels.npy")

        all_X.append(X)
        all_y.append(y)

        if verbose:
            print(f"   Loaded subject {subject_id:02d}: {X.shape}")

    # Concatenate all data
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    if verbose:
        print(f"\n   Total data shape: {X_all.shape}")
        print(f"   Total labels: {len(y_all)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all  # Maintain class distribution
    )

    # Save splits
    np.save(data_dir / "train_data.npy", X_train)
    np.save(data_dir / "train_labels.npy", y_train)
    np.save(data_dir / "test_data.npy", X_test)
    np.save(data_dir / "test_labels.npy", y_test)

    if verbose:
        print(f"\n✅ Split created:")
        print(f"   Train: {X_train.shape} - {len(y_train)} trials")
        print(f"   Test:  {X_test.shape} - {len(y_test)} trials")
        print(f"   Train class distribution: {np.bincount(y_train)}")
        print(f"   Test class distribution: {np.bincount(y_test)}")
        print(f"   Saved to: {data_dir}")


def main():
    """
    Main function for command-line usage.

    Usage:
        python preprocessing.py                # Process all subjects
        python preprocessing.py --subjects 1 2 # Process specific subjects
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess BCI Competition IV-2a dataset"
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=None,
        help='Subject IDs to process (1-9). Default: all'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save processed data. Default: data/processed/'
    )
    parser.add_argument(
        '--no_filter',
        action='store_true',
        help='Skip bandpass filtering'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Skip normalization'
    )
    parser.add_argument(
        '--create_split',
        action='store_true',
        help='Create train/test split after processing'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = EEGPreprocessor()

    # Process subjects
    info = preprocessor.process_all_subjects(
        subjects=args.subjects,
        save_dir=args.save_dir
    )

    # Create train/test split if requested
    if args.create_split:
        create_train_test_split(data_dir=args.save_dir)

    print("\n✨ Next steps:")
    print("  1. Explore data: jupyter lab notebooks/01_explore_data.ipynb")
    print("  2. Train model: python src/training/train.py")


if __name__ == "__main__":
    main()
