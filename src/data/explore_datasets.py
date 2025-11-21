"""
Explore Available Motor Imagery Datasets
=========================================

Check what datasets are available in MOABB for motor imagery.

Author: Temur Turayev
TashPMI, 2024
"""

from moabb import datasets
from moabb.paradigms import MotorImagery

def list_available_datasets():
    """List all available motor imagery datasets in MOABB."""

    print("=" * 70)
    print("🔍 AVAILABLE MOTOR IMAGERY DATASETS IN MOABB")
    print("=" * 70)

    # Get all dataset classes
    all_datasets = [d for d in dir(datasets) if not d.startswith('_')]

    motor_imagery_datasets = []

    for dataset_name in all_datasets:
        try:
            dataset_class = getattr(datasets, dataset_name)

            # Check if it's a class (not a module or function)
            if not isinstance(dataset_class, type):
                continue

            # Try to instantiate
            try:
                dataset = dataset_class()

                # Check if it has motor imagery events
                if hasattr(dataset, 'event_id'):
                    events = dataset.event_id

                    # Check for motor imagery keywords
                    mi_keywords = ['hand', 'feet', 'tongue', 'left', 'right', 'motor', 'imagery']
                    is_mi = any(keyword in str(events).lower() for keyword in mi_keywords)

                    if is_mi:
                        n_subjects = len(dataset.subject_list) if hasattr(dataset, 'subject_list') else 'Unknown'
                        n_sessions = dataset.n_sessions if hasattr(dataset, 'n_sessions') else 'Unknown'

                        motor_imagery_datasets.append({
                            'name': dataset_name,
                            'subjects': n_subjects,
                            'sessions': n_sessions,
                            'events': events,
                            'dataset': dataset
                        })

            except Exception as e:
                continue

        except Exception as e:
            continue

    # Print results
    print(f"\nFound {len(motor_imagery_datasets)} motor imagery datasets:\n")

    for i, ds in enumerate(motor_imagery_datasets, 1):
        print(f"{i}. {ds['name']}")
        print(f"   Subjects: {ds['subjects']}")
        print(f"   Sessions: {ds['sessions']}")
        print(f"   Events: {ds['events']}")
        print()

    return motor_imagery_datasets


def get_dataset_details(dataset_name: str):
    """Get detailed information about a specific dataset."""

    print("=" * 70)
    print(f"📊 DATASET DETAILS: {dataset_name}")
    print("=" * 70)

    try:
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class()

        print(f"\n📝 Basic Information:")
        print(f"   Name: {dataset_name}")
        print(f"   Subjects: {len(dataset.subject_list) if hasattr(dataset, 'subject_list') else 'Unknown'}")
        print(f"   Sessions: {dataset.n_sessions if hasattr(dataset, 'n_sessions') else 'Unknown'}")

        if hasattr(dataset, 'event_id'):
            print(f"   Events: {dataset.event_id}")

        # Try to get paradigm info
        try:
            paradigm = MotorImagery(n_classes=len(dataset.event_id))
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]])

            print(f"\n📊 Data Shape:")
            print(f"   Trials: {X.shape[0]}")
            print(f"   Channels: {X.shape[1]}")
            print(f"   Samples: {X.shape[2]}")
            print(f"   Sampling Rate: {meta.iloc[0]['sfreq']} Hz")

            print(f"\n🎯 Class Distribution:")
            import numpy as np
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"   Class {label}: {count} trials")

        except Exception as e:
            print(f"\n⚠️  Could not load sample data: {e}")

        return dataset

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        return None


if __name__ == "__main__":
    # List all available datasets
    datasets_list = list_available_datasets()

    # Get details for key datasets
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDED DATASETS FOR NEUROHAND")
    print("=" * 70)

    recommended = [
        'BNCI2014_001',  # Already using
        'BNCI2014_004',  # Similar to 2014_001
        'PhysionetMI',   # 109 subjects!
        'Cho2017',       # 52 subjects
        'Weibo2014',     # 10 subjects
    ]

    for ds_name in recommended:
        if any(d['name'] == ds_name for d in datasets_list):
            print(f"\n✅ {ds_name} - Available")
        else:
            print(f"\n❌ {ds_name} - Not found")
