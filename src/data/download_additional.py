"""
Download Additional Motor Imagery Datasets
===========================================

Download and preprocess additional datasets to expand training data:
- BNCI2014_004 (9 subjects × 5 sessions, 2 classes)
- Cho2017 (52 subjects, 4 classes)
- PhysionetMI (109 subjects, 4 classes)

Author: Temur Turayev
TashPMI, 2024
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from moabb.datasets import BNCI2014_004, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
from scipy.signal import butter, filtfilt
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def bandpass_filter(data, lowcut=4.0, highcut=38.0, fs=250.0, order=5):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def standardize(data):
    """Per-channel standardization."""
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)


def map_labels_to_4class(labels, dataset_name):
    """
    Map dataset labels to 4-class standard (0=left, 1=right, 2=feet, 3=tongue).

    Some datasets only have 2 classes (left/right), so we mark them separately.
    """
    if dataset_name == 'BNCI2014_004':
        # Only left/right hand (classes 0, 1)
        return labels, 2  # 2 classes

    elif dataset_name in ['Cho2017', 'PhysionetMI']:
        # 4 classes (left, right, feet, rest/tongue)
        # These datasets typically have: left_hand, right_hand, feet, rest
        # We'll map rest → tongue for consistency
        return labels, 4  # 4 classes

    else:
        return labels, 4


def download_dataset(dataset_class, dataset_name, n_classes=4):
    """
    Download and preprocess a single dataset.

    Args:
        dataset_class: MOABB dataset class
        dataset_name: Name for logging
        n_classes: Number of classes (2 or 4)

    Returns:
        data, labels, metadata
    """
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING: {dataset_name}")
    print(f"{'='*70}")

    try:
        # Initialize dataset
        dataset = dataset_class()

        print(f"✅ Dataset initialized")
        print(f"   Subjects: {len(dataset.subject_list)}")
        print(f"   Sessions: {dataset.n_sessions}")

        # Create paradigm
        paradigm = MotorImagery(
            events=['left_hand', 'right_hand', 'feet', 'tongue'] if n_classes == 4
                   else ['left_hand', 'right_hand'],
            n_classes=n_classes,
            fmin=4,
            fmax=38,
            resample=250
        )

        print(f"\n📊 Downloading data for all subjects...")

        all_data = []
        all_labels = []
        all_meta = []

        # Download all subjects
        for subject in tqdm(dataset.subject_list, desc='Subjects'):
            try:
                X, labels, meta = paradigm.get_data(
                    dataset=dataset,
                    subjects=[subject],
                    return_epochs=False
                )

                all_data.append(X)
                all_labels.append(labels)
                all_meta.append(meta)

            except Exception as e:
                print(f"   ⚠️ Subject {subject} failed: {e}")
                continue

        if not all_data:
            print(f"   ❌ No data downloaded for {dataset_name}")
            return None, None, None

        # Concatenate all subjects
        X_all = np.concatenate(all_data, axis=0)
        y_all = np.concatenate(all_labels, axis=0)

        print(f"\n✅ Download complete!")
        print(f"   Total trials: {X_all.shape[0]}")
        print(f"   Channels: {X_all.shape[1]}")
        print(f"   Samples: {X_all.shape[2]}")
        print(f"   Classes: {len(np.unique(y_all))}")

        # Class distribution
        unique, counts = np.unique(y_all, return_counts=True)
        print(f"\n📊 Class distribution:")
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue/Rest']
        for label, count in zip(unique, counts):
            name = class_names[int(label)] if int(label) < len(class_names) else f"Class {label}"
            print(f"   {name}: {count} trials")

        return X_all, y_all, {'dataset': dataset_name, 'n_classes': n_classes}

    except Exception as e:
        print(f"\n❌ Error downloading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def preprocess_dataset(X, y, metadata):
    """
    Preprocess dataset with bandpass filtering and standardization.

    Args:
        X: EEG data [trials, channels, samples]
        y: Labels [trials]
        metadata: Dataset metadata

    Returns:
        Preprocessed X, y
    """
    print(f"\n🔧 Preprocessing {metadata['dataset']}...")

    # Bandpass filter
    print(f"   Filtering (4-38 Hz)...")
    X_filtered = bandpass_filter(X, lowcut=4.0, highcut=38.0, fs=250.0)

    # Standardize
    print(f"   Standardizing...")
    X_standardized = standardize(X_filtered)

    # Ensure correct shape for 4-class model
    # If dataset has only 2 classes, we'll save it separately

    print(f"✅ Preprocessing complete")
    print(f"   Shape: {X_standardized.shape}")

    return X_standardized, y


def main():
    """Main function to download all datasets."""

    print("="*70)
    print("🚀 DOWNLOADING ADDITIONAL MOTOR IMAGERY DATASETS")
    print("="*70)
    print("\nTarget datasets:")
    print("  1. BNCI2014_004 (9 subjects, 2 classes: left/right)")
    print("  2. Cho2017 (52 subjects, 4 classes)")
    print("  3. PhysionetMI (109 subjects, 4 classes)")
    print("\nThis will take 30-60 minutes depending on internet speed...")

    # Create output directory
    output_dir = PROJECT_ROOT / 'data' / 'additional'
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = [
        {
            'class': BNCI2014_004,
            'name': 'BNCI2014_004',
            'n_classes': 2,
            'priority': 'high'  # Similar to our baseline dataset
        },
        {
            'class': Cho2017,
            'name': 'Cho2017',
            'n_classes': 4,
            'priority': 'high'  # Large dataset with 4 classes
        },
        {
            'class': PhysionetMI,
            'name': 'PhysionetMI',
            'n_classes': 4,
            'priority': 'medium'  # Huge dataset, might take long
        },
    ]

    successful_downloads = []

    for dataset_info in datasets_to_download:
        print(f"\n\n{'='*70}")
        print(f"Processing: {dataset_info['name']} (Priority: {dataset_info['priority']})")
        print(f"{'='*70}")

        # Download
        X, y, metadata = download_dataset(
            dataset_info['class'],
            dataset_info['name'],
            dataset_info['n_classes']
        )

        if X is None:
            print(f"   ⚠️ Skipping {dataset_info['name']}")
            continue

        # Preprocess
        X_proc, y_proc = preprocess_dataset(X, y, metadata)

        # Save
        save_path = output_dir / f"{dataset_info['name'].lower()}"
        save_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving to {save_path}...")
        np.save(save_path / 'data.npy', X_proc)
        np.save(save_path / 'labels.npy', y_proc)

        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✅ Saved!")

        successful_downloads.append({
            'name': dataset_info['name'],
            'trials': X_proc.shape[0],
            'classes': dataset_info['n_classes'],
            'path': save_path
        })

    # Summary
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)

    if successful_downloads:
        print(f"\nSuccessfully downloaded {len(successful_downloads)} datasets:\n")

        total_trials = 0
        for ds in successful_downloads:
            print(f"✅ {ds['name']}")
            print(f"   Trials: {ds['trials']}")
            print(f"   Classes: {ds['classes']}")
            print(f"   Path: {ds['path']}")
            print()
            total_trials += ds['trials']

        print(f"📊 Total new trials: {total_trials:,}")
        print(f"📊 Original trials: 5,184")
        print(f"📊 Combined total: {5184 + total_trials:,}")

        # Save summary
        summary_path = output_dir / 'download_summary.pkl'
        with open(summary_path, 'wb') as f:
            pickle.dump(successful_downloads, f)

        print(f"\n💾 Summary saved to: {summary_path}")

    else:
        print("\n❌ No datasets downloaded successfully")

    print("\n🎯 Next steps:")
    print("  1. Check downloaded data in data/additional/")
    print("  2. Combine with original dataset")
    print("  3. Train improved model on expanded dataset")


if __name__ == "__main__":
    main()
