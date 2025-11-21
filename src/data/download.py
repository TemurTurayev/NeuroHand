"""
BCI Competition IV Dataset 2a Downloader
==========================================

Автоматическая загрузка датасета BCI Competition IV-2a для обучения EEGNet.

Dataset Details:
- 9 subjects (A01-A09)
- 2 sessions per subject
- 288 trials per session (72 per class)
- 4 classes: Left Hand, Right Hand, Feet, Tongue
- 22 EEG channels + 3 EOG channels
- Sampling rate: 250 Hz
- Trial length: 4 seconds (1000 samples)

Автор: Temur Turayev
TashPMI, 2024
"""

import os
import sys
from pathlib import Path
from typing import Optional

import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class BCIDataDownloader:
    """
    Класс для загрузки и базовой обработки BCI Competition IV-2a dataset.

    Uses MOABB (Mother of All BCI Benchmarks) library for standardized access.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize downloader.

        Args:
            data_dir: Directory to save raw data (default: PROJECT_ROOT/data/raw)
            verbose: Print progress messages
        """
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / "data" / "raw"
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        if self.verbose:
            print(f"📁 Data directory: {self.data_dir}")

    def download(self, subjects: Optional[list] = None) -> dict:
        """
        Download BCI Competition IV-2a dataset.

        Args:
            subjects: List of subject IDs (1-9). If None, downloads all subjects.

        Returns:
            Dictionary with dataset information

        Examples:
            >>> downloader = BCIDataDownloader()
            >>> info = downloader.download(subjects=[1, 2])  # Download first 2 subjects
            >>> info = downloader.download()  # Download all 9 subjects
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🧠 BCI Competition IV Dataset 2a Downloader")
            print("="*60)

        # Initialize MOABB dataset
        # BNCI2014_001 is the MOABB name for BCI Competition IV-2a
        dataset = BNCI2014_001()

        # Get subject list
        if subjects is None:
            subjects = list(range(1, 10))  # All 9 subjects

        if self.verbose:
            print(f"\n📊 Dataset Info:")
            print(f"  - Name: {dataset.code}")
            print(f"  - Subjects: {len(subjects)}")
            print(f"  - Paradigm: Motor Imagery")
            print(f"  - Classes: 4 (Left Hand, Right Hand, Feet, Tongue)")
            print(f"  - Sampling Rate: 250 Hz")
            print(f"  - Channels: 22 EEG + 3 EOG")

        # Download data for each subject
        all_data = {}

        if self.verbose:
            print(f"\n⬇️  Downloading data...")
            subject_iter = tqdm(subjects, desc="Subjects")
        else:
            subject_iter = subjects

        for subject_id in subject_iter:
            try:
                # Load data for this subject
                # MOABB automatically downloads and caches data
                sessions = dataset.get_data(subjects=[subject_id])

                all_data[subject_id] = sessions[subject_id]

                if self.verbose:
                    n_sessions = len(sessions[subject_id])
                    n_runs = sum(len(sess) for sess in sessions[subject_id].values())
                    print(f"  ✅ Subject {subject_id:02d}: {n_sessions} sessions, {n_runs} runs")

            except Exception as e:
                print(f"  ❌ Error downloading subject {subject_id}: {e}")
                continue

        # Create dataset info
        info = {
            'dataset_name': 'BCI Competition IV-2a (BNCI2014_001)',
            'n_subjects': len(all_data),
            'subjects': list(all_data.keys()),
            'n_classes': 4,
            'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
            'sampling_rate': 250,
            'n_channels': 22,  # EEG only (excluding EOG)
            'trial_length': 4.0,  # seconds
            'data_dir': str(self.data_dir)
        }

        if self.verbose:
            print(f"\n✅ Download complete!")
            print(f"   Total subjects downloaded: {info['n_subjects']}")
            print(f"   Data cached by MOABB in: ~/mne_data/")

        return info

    def get_subject_data(self, subject_id: int) -> dict:
        """
        Get raw MNE data for a specific subject.

        Args:
            subject_id: Subject number (1-9)

        Returns:
            Dictionary with sessions and runs data
        """
        dataset = BNCI2014_001()
        data = dataset.get_data(subjects=[subject_id])
        return data[subject_id]

    def verify_download(self) -> bool:
        """
        Verify that data was downloaded correctly.

        Returns:
            True if data exists, False otherwise
        """
        try:
            dataset = BNCI2014_001()
            # Try to load one subject to verify
            test_data = dataset.get_data(subjects=[1])

            if self.verbose:
                print("✅ Data verification successful!")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Data verification failed: {e}")
            return False


def main():
    """
    Main function for command-line usage.

    Usage:
        python download.py                    # Download all subjects
        python download.py --subjects 1 2 3   # Download specific subjects
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download BCI Competition IV-2a dataset"
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=None,
        help='Subject IDs to download (1-9). Default: all'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory to save data. Default: data/raw/'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify download after completion'
    )

    args = parser.parse_args()

    # Create downloader
    downloader = BCIDataDownloader(data_dir=args.data_dir)

    # Download data
    info = downloader.download(subjects=args.subjects)

    # Verify if requested
    if args.verify:
        print("\n🔍 Verifying download...")
        success = downloader.verify_download()
        if not success:
            sys.exit(1)

    # Print summary
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("="*60)

    print("\n✨ Next steps:")
    print("  1. Run: python src/data/preprocessing.py")
    print("  2. Then: python src/training/train.py")
    print("  3. Or explore: jupyter lab notebooks/01_explore_data.ipynb")


if __name__ == "__main__":
    main()
