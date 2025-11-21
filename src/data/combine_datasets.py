"""
Combine Multiple Motor Imagery Datasets
========================================

Combine baseline BNCI2014_001 with additional datasets:
- BNCI2014_004 (9 subjects, 2 classes)
- Cho2017 (52 subjects, 4 classes)
- PhysionetMI (109 subjects, 4 classes)

Strategy:
- For 2-class datasets: Only use trials that match left/right hand
- For 4-class datasets: Use all trials
- Ensure consistent preprocessing and label mapping

Author: Temur Turayev
TashPMI, 2024
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_dataset(dataset_path, dataset_name):
    """Load a single dataset from disk."""
    print(f"\n📂 Loading {dataset_name}...")

    data_path = Path(dataset_path)

    if not data_path.exists():
        print(f"   ⚠️ Path not found: {data_path}")
        return None, None, None

    # Load data
    X = np.load(data_path / 'data.npy')
    y = np.load(data_path / 'labels.npy')

    with open(data_path / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"   ✅ Loaded {len(X)} trials")
    print(f"      Shape: {X.shape}")
    print(f"      Classes: {metadata.get('n_classes', 'unknown')}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue/Rest']
    print(f"      Distribution:")
    for label, count in zip(unique, counts):
        name = class_names[int(label)] if int(label) < len(class_names) else f"Class {label}"
        print(f"        {name}: {count}")

    return X, y, metadata


def filter_2class_dataset(X, y, keep_classes=[0, 1]):
    """
    Filter 2-class datasets to only keep specific classes.

    Args:
        X: Data [trials, channels, samples]
        y: Labels [trials]
        keep_classes: List of class IDs to keep (default: [0, 1] = left/right hand)

    Returns:
        Filtered X, y
    """
    mask = np.isin(y, keep_classes)
    return X[mask], y[mask]


def combine_datasets(baseline_path, additional_datasets):
    """
    Combine baseline dataset with additional datasets.

    Args:
        baseline_path: Path to baseline BNCI2014_001 dataset
        additional_datasets: List of paths to additional datasets

    Returns:
        Combined X_train, X_test, y_train, y_test
    """
    print("="*70)
    print("🔗 COMBINING DATASETS")
    print("="*70)

    # Load baseline
    print("\n1️⃣ Loading baseline dataset (BNCI2014_001)...")
    X_base_train = np.load(baseline_path / 'train_data.npy')
    y_base_train = np.load(baseline_path / 'train_labels.npy')
    X_base_test = np.load(baseline_path / 'test_data.npy')
    y_base_test = np.load(baseline_path / 'test_labels.npy')

    print(f"   ✅ Baseline loaded")
    print(f"      Train: {len(X_base_train)} trials")
    print(f"      Test: {len(X_base_test)} trials")
    print(f"      Shape: {X_base_train.shape}")

    # Initialize lists with baseline
    train_datasets = [(X_base_train, y_base_train, 'BNCI2014_001')]
    test_datasets = [(X_base_test, y_base_test, 'BNCI2014_001')]

    # Load additional datasets
    print("\n2️⃣ Loading additional datasets...")

    for dataset_info in additional_datasets:
        X, y, metadata = load_dataset(
            dataset_info['path'],
            dataset_info['name']
        )

        if X is None:
            print(f"   ⚠️ Skipping {dataset_info['name']}")
            continue

        # Check if dataset needs filtering
        if metadata.get('n_classes') == 2:
            print(f"   🔧 Filtering 2-class dataset (keeping left/right only)...")
            X, y = filter_2class_dataset(X, y, keep_classes=[0, 1])
            print(f"      Remaining trials: {len(X)}")

        # Check shape compatibility
        if X.shape[1] != X_base_train.shape[1]:
            print(f"   ⚠️ Incompatible number of channels: {X.shape[1]} vs {X_base_train.shape[1]}")
            print(f"   ⚠️ Skipping {dataset_info['name']}")
            continue

        if X.shape[2] != X_base_train.shape[2]:
            print(f"   ⚠️ Incompatible sample length: {X.shape[2]} vs {X_base_train.shape[2]}")
            print(f"   ⚠️ Skipping {dataset_info['name']}")
            continue

        # Split into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        train_datasets.append((X_train, y_train, dataset_info['name']))
        test_datasets.append((X_test, y_test, dataset_info['name']))

        print(f"   ✅ Added {dataset_info['name']}")
        print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # Combine all datasets
    print("\n3️⃣ Combining all datasets...")

    X_train_combined = np.concatenate([X for X, _, _ in train_datasets], axis=0)
    y_train_combined = np.concatenate([y for _, y, _ in train_datasets], axis=0)

    X_test_combined = np.concatenate([X for X, _, _ in test_datasets], axis=0)
    y_test_combined = np.concatenate([y for _, y, _ in test_datasets], axis=0)

    print(f"   ✅ Combined!")
    print(f"\n📊 Final dataset statistics:")
    print(f"   Training samples: {len(X_train_combined):,}")
    print(f"   Test samples: {len(X_test_combined):,}")
    print(f"   Total samples: {len(X_train_combined) + len(X_test_combined):,}")
    print(f"   Shape: {X_train_combined.shape}")

    # Class distribution
    print(f"\n📊 Training class distribution:")
    unique, counts = np.unique(y_train_combined, return_counts=True)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    for label, count in zip(unique, counts):
        name = class_names[int(label)]
        pct = 100 * count / len(y_train_combined)
        print(f"   {name}: {count:,} ({pct:.1f}%)")

    # Per-dataset breakdown
    print(f"\n📊 Dataset contributions:")
    for X, y, name in train_datasets:
        pct = 100 * len(X) / len(X_train_combined)
        print(f"   {name}: {len(X):,} trials ({pct:.1f}%)")

    return X_train_combined, X_test_combined, y_train_combined, y_test_combined


def main():
    """Main function to combine datasets."""

    print("="*70)
    print("🚀 DATASET COMBINATION")
    print("="*70)

    # Paths
    baseline_path = PROJECT_ROOT / 'data' / 'processed'
    additional_path = PROJECT_ROOT / 'data' / 'additional'
    output_path = PROJECT_ROOT / 'data' / 'combined'
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if baseline exists
    if not baseline_path.exists():
        print(f"\n❌ Baseline dataset not found at: {baseline_path}")
        print(f"   Please run preprocessing first!")
        return

    # Check for additional datasets
    additional_datasets = []

    for dataset_name in ['bnci2014_004', 'cho2017', 'physionetmi']:
        dataset_path = additional_path / dataset_name
        if dataset_path.exists():
            additional_datasets.append({
                'name': dataset_name.upper(),
                'path': dataset_path
            })

    if not additional_datasets:
        print(f"\n⚠️ No additional datasets found in: {additional_path}")
        print(f"   Please run download_additional.py first!")
        print(f"\nFalling back to baseline dataset only...")

        # Just copy baseline
        X_train = np.load(baseline_path / 'train_data.npy')
        y_train = np.load(baseline_path / 'train_labels.npy')
        X_test = np.load(baseline_path / 'test_data.npy')
        y_test = np.load(baseline_path / 'test_labels.npy')

    else:
        print(f"\n✅ Found {len(additional_datasets)} additional datasets:")
        for ds in additional_datasets:
            print(f"   - {ds['name']}")

        # Combine datasets
        X_train, X_test, y_train, y_test = combine_datasets(
            baseline_path,
            additional_datasets
        )

    # Save combined dataset
    print(f"\n💾 Saving combined dataset to: {output_path}")

    np.save(output_path / 'train_data.npy', X_train)
    np.save(output_path / 'train_labels.npy', y_train)
    np.save(output_path / 'test_data.npy', X_test)
    np.save(output_path / 'test_labels.npy', y_test)

    # Save metadata
    metadata = {
        'total_train_samples': len(X_train),
        'total_test_samples': len(X_test),
        'shape': X_train.shape,
        'datasets': [ds['name'] for ds in additional_datasets] + ['BNCI2014_001'],
        'created_date': str(np.datetime64('now'))
    }

    with open(output_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"   ✅ Saved!")

    print("\n" + "="*70)
    print("✅ DATASET COMBINATION COMPLETE!")
    print("="*70)

    print(f"\n📊 Summary:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Total increase from baseline: +{len(X_train) - 4147:,} train, +{len(X_test) - 1037:,} test")

    print(f"\n🎯 Next step:")
    print(f"   Train improved model:")
    print(f"   python src/training/train_improved.py --data_dir data/combined --epochs 500")


if __name__ == "__main__":
    main()
