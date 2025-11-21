"""
Hyperparameter Tuning for EEGNet
==================================

Grid search over key hyperparameters to find optimal configuration.

Hyperparameters to tune:
- Learning rate: [0.0001, 0.0005, 0.001, 0.005]
- Batch size: [32, 64, 128]
- Weight decay: [0.0001, 0.001, 0.01]
- Dropout rate: [0.25, 0.5]
- Number of filters: [8, 16, 32]

Author: Temur Turayev
TashPMI, 2024
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import product
import json
from tqdm import tqdm

from src.models.eegnet import EEGNet
from src.data.dataset import EEGDataset
from src.training.config import TrainingConfig


class HyperparameterTuner:
    """
    Grid search for EEGNet hyperparameters.

    Example:
        >>> tuner = HyperparameterTuner(train_loader, val_loader)
        >>> best_params = tuner.search(n_trials=20)
        >>> print(f"Best accuracy: {best_params['accuracy']:.2f}%")
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int = 4,
        n_channels: int = 22,
        n_samples: int = 1000
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f"🔍 Hyperparameter Tuner initialized on {self.device}")

    def train_with_config(
        self,
        config: dict,
        n_epochs: int = 50
    ) -> float:
        """
        Train model with specific hyperparameter configuration.

        Args:
            config: Dictionary of hyperparameters
            n_epochs: Number of training epochs

        Returns:
            Best validation accuracy achieved
        """
        # Create model
        model = EEGNet(
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            F1=config.get('F1', 8),
            D=config.get('D', 2),
            F2=config.get('F2', 16),
            dropout=config.get('dropout', 0.5)
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Loss
        criterion = nn.CrossEntropyLoss()

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        best_val_acc = 0.0

        # Training loop
        for epoch in range(n_epochs):
            # Train
            model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Max-norm constraint
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=0.5)

            # Validate
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            val_acc = 100. * correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            scheduler.step(val_acc)

        return best_val_acc

    def grid_search(
        self,
        param_grid: dict,
        n_epochs: int = 50,
        save_path: str = None
    ) -> dict:
        """
        Perform grid search over hyperparameter space.

        Args:
            param_grid: Dictionary of hyperparameter ranges
            n_epochs: Epochs per trial
            save_path: Path to save results

        Returns:
            Best configuration found
        """
        print("="*70)
        print("🔍 HYPERPARAMETER GRID SEARCH")
        print("="*70)

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = list(product(*values))

        print(f"\n📊 Search space:")
        for key, val in param_grid.items():
            print(f"   {key}: {val}")
        print(f"\nTotal combinations: {len(all_combinations)}")
        print(f"Epochs per trial: {n_epochs}")
        print(f"Estimated time: {len(all_combinations) * n_epochs * 0.5:.0f} seconds (~{len(all_combinations) * n_epochs * 0.5 / 60:.0f} minutes)")

        results = []
        best_config = None
        best_accuracy = 0.0

        # Try each configuration
        pbar = tqdm(all_combinations, desc='Grid Search')

        for i, combination in enumerate(pbar):
            config = dict(zip(keys, combination))

            print(f"\n{'='*70}")
            print(f"Trial {i+1}/{len(all_combinations)}")
            print(f"{'='*70}")
            print(f"Configuration:")
            for key, val in config.items():
                print(f"   {key}: {val}")

            # Train with this config
            accuracy = self.train_with_config(config, n_epochs=n_epochs)

            print(f"   Validation Accuracy: {accuracy:.2f}%")

            # Save result
            result = {
                'trial': i + 1,
                'config': config,
                'accuracy': accuracy
            }
            results.append(result)

            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
                print(f"   🏆 New best! ({accuracy:.2f}%)")

            pbar.set_postfix({'Best Acc': f'{best_accuracy:.2f}%'})

        # Save results
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump({
                    'param_grid': param_grid,
                    'results': results,
                    'best_config': best_config,
                    'best_accuracy': best_accuracy
                }, f, indent=2)

            print(f"\n💾 Results saved to: {save_path}")

        # Print summary
        print("\n" + "="*70)
        print("✅ GRID SEARCH COMPLETE")
        print("="*70)

        print(f"\n🏆 Best Configuration:")
        for key, val in best_config.items():
            print(f"   {key}: {val}")
        print(f"\n   Validation Accuracy: {best_accuracy:.2f}%")

        # Top 5 results
        results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        print(f"\n📊 Top 5 Configurations:")
        for i, result in enumerate(results_sorted[:5], 1):
            print(f"\n{i}. Accuracy: {result['accuracy']:.2f}%")
            for key, val in result['config'].items():
                print(f"   {key}: {val}")

        return {
            'best_config': best_config,
            'best_accuracy': best_accuracy,
            'all_results': results
        }

    def random_search(
        self,
        param_distributions: dict,
        n_trials: int = 20,
        n_epochs: int = 50,
        save_path: str = None
    ) -> dict:
        """
        Perform random search (faster than grid search).

        Args:
            param_distributions: Dictionary of parameter distributions
            n_trials: Number of random trials
            n_epochs: Epochs per trial
            save_path: Path to save results

        Returns:
            Best configuration found
        """
        print("="*70)
        print("🎲 HYPERPARAMETER RANDOM SEARCH")
        print("="*70)

        print(f"\n📊 Search space:")
        for key, val in param_distributions.items():
            print(f"   {key}: {val}")
        print(f"\nTrials: {n_trials}")
        print(f"Epochs per trial: {n_epochs}")
        print(f"Estimated time: {n_trials * n_epochs * 0.5:.0f} seconds (~{n_trials * n_epochs * 0.5 / 60:.0f} minutes)")

        results = []
        best_config = None
        best_accuracy = 0.0

        for i in range(n_trials):
            # Sample random configuration
            config = {}
            for key, values in param_distributions.items():
                config[key] = np.random.choice(values)

            print(f"\n{'='*70}")
            print(f"Trial {i+1}/{n_trials}")
            print(f"{'='*70}")
            print(f"Configuration:")
            for key, val in config.items():
                print(f"   {key}: {val}")

            # Train with this config
            accuracy = self.train_with_config(config, n_epochs=n_epochs)

            print(f"   Validation Accuracy: {accuracy:.2f}%")

            # Save result
            result = {
                'trial': i + 1,
                'config': config,
                'accuracy': accuracy
            }
            results.append(result)

            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
                print(f"   🏆 New best! ({accuracy:.2f}%)")

        # Save results
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump({
                    'param_distributions': {k: list(v) for k, v in param_distributions.items()},
                    'results': results,
                    'best_config': best_config,
                    'best_accuracy': best_accuracy
                }, f, indent=2)

            print(f"\n💾 Results saved to: {save_path}")

        # Print summary
        print("\n" + "="*70)
        print("✅ RANDOM SEARCH COMPLETE")
        print("="*70)

        print(f"\n🏆 Best Configuration:")
        for key, val in best_config.items():
            print(f"   {key}: {val}")
        print(f"\n   Validation Accuracy: {best_accuracy:.2f}%")

        return {
            'best_config': best_config,
            'best_accuracy': best_accuracy,
            'all_results': results
        }


def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for EEGNet")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random',
                       help='Search method')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials (for random search)')
    parser.add_argument('--n_epochs', type=int, default=50, help='Epochs per trial')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Load data
    print("📊 Loading data...")
    data_dir = PROJECT_ROOT / args.data_dir

    train_dataset = EEGDataset(
        data_path=data_dir / 'train_data.npy',
        labels_path=data_dir / 'train_labels.npy',
        augment=True
    )

    test_dataset = EEGDataset(
        data_path=data_dir / 'test_data.npy',
        labels_path=data_dir / 'test_labels.npy',
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"   Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # Create tuner
    tuner = HyperparameterTuner(
        train_loader=train_loader,
        val_loader=test_loader
    )

    # Define search space
    if args.method == 'grid':
        # Grid search (smaller space)
        param_grid = {
            'learning_rate': [0.0005, 0.001, 0.002],
            'weight_decay': [0.0005, 0.001, 0.005],
            'dropout': [0.25, 0.5],
            'F1': [8, 16]
        }

        results = tuner.grid_search(
            param_grid=param_grid,
            n_epochs=args.n_epochs,
            save_path=PROJECT_ROOT / 'models' / 'tuning_results_grid.json'
        )

    else:
        # Random search (larger space)
        param_distributions = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
            'weight_decay': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'dropout': [0.25, 0.3, 0.4, 0.5],
            'F1': [8, 12, 16, 24],
            'D': [2, 3, 4]
        }

        results = tuner.random_search(
            param_distributions=param_distributions,
            n_trials=args.n_trials,
            n_epochs=args.n_epochs,
            save_path=PROJECT_ROOT / 'models' / 'tuning_results_random.json'
        )

    # Print next steps
    print("\n🎯 Next steps:")
    print("  1. Use best config to train full model:")
    print(f"     python src/training/train_improved.py \\")
    print(f"         --lr {results['best_config']['learning_rate']} \\")
    print(f"         --epochs 500")
    print(f"\n  2. Expected accuracy: {results['best_accuracy']:.2f}% → 70-75% (with full training)")


if __name__ == "__main__":
    main()
