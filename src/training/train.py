"""
Training Script for EEGNet
============================

Полный pipeline для обучения EEGNet на motor imagery data.

Автор: Temur Turayev
TashPMI, 2024
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.eegnet import EEGNet
from src.models.utils import save_checkpoint, count_parameters
from src.data.dataset import create_data_loaders
from src.training.config import TrainingConfig


class Trainer:
    """
    Trainer class для обучения EEGNet.

    Handles:
        - Training loop
        - Validation
        - Early stopping
        - Checkpoint saving
        - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig
    ):
        """
        Initialize trainer.

        Args:
            model: EEGNet model
            train_loader: Training data loader
            test_loader: Test data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # Set device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Loss function (CrossEntropy for classification)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (Adam - adaptive learning rate)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler (reduce LR when loss plateaus)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rates': []
        }

        # Early stopping
        self.best_test_acc = 0.0
        self.epochs_no_improve = 0

        if config.verbose:
            print(f"\n🚀 Trainer initialized")
            print(f"   Device: {self.device}")
            print(f"   Model parameters: {count_parameters(model):,}")
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Test batches: {len(test_loader)}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average loss and accuracy for this epoch

        Training Process:
            1. Set model to training mode
            2. Iterate through batches
            3. Forward pass (compute predictions)
            4. Compute loss
            5. Backward pass (compute gradients)
            6. Update weights
            7. Apply max norm constraint
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]",
            disable=not self.config.verbose
        )

        for batch_idx, (signals, labels) in enumerate(pbar):
            # Move data to device
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(signals)  # [batch, n_classes]

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Apply max norm constraint (EEGNet specific)
            self.model.apply_max_norm_constraint()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            # Update statistics
            total_loss += loss.item() * signals.size(0)
            total_correct += correct
            total_samples += signals.size(0)

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                current_acc = 100.0 * correct / signals.size(0)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Tuple[float, float]:
        """
        Evaluate on test set.

        Args:
            epoch: Current epoch number

        Returns:
            Average loss and accuracy on test set

        Evaluation Process:
            1. Set model to evaluation mode (disables dropout)
            2. Iterate through test batches
            3. Compute predictions (no gradient computation)
            4. Calculate metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(
            self.test_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs} [Test]",
            disable=not self.config.verbose
        )

        for signals, labels in pbar:
            # Move data to device
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (no gradient computation)
            outputs = self.model(signals)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            # Update statistics
            total_loss += loss.item() * signals.size(0)
            total_correct += correct
            total_samples += signals.size(0)

        # Calculate metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        return avg_loss, avg_acc

    def train(self) -> Dict:
        """
        Full training loop.

        Returns:
            Training history dictionary

        Training Loop:
            For each epoch:
                1. Train on training set
                2. Evaluate on test set
                3. Update learning rate if needed
                4. Save checkpoint if best model
                5. Check early stopping
        """
        print("\n" + "="*70)
        print("🎯 TRAINING STARTED")
        print("="*70)
        print(f"Configuration:")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Device: {self.device}")
        print("="*70 + "\n")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # Evaluate
            test_loss, test_acc = self.evaluate(epoch)

            # Update learning rate scheduler
            self.scheduler.step(test_loss)

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            if self.config.verbose:
                print(f"\nEpoch {epoch+1}/{self.config.epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
                print(f"  LR: {current_lr:.6f}")

            # Save best model
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.epochs_no_improve = 0

                if self.config.save_best_only:
                    checkpoint_path = self.config.save_dir / "best_model.pth"
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=test_loss,
                        accuracy=test_acc,
                        filepath=str(checkpoint_path)
                    )
                    if self.config.verbose:
                        print(f"  ✅ Best model saved! (Test Acc: {test_acc:.2f}%)")
            else:
                self.epochs_no_improve += 1

            # Early stopping check
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered!")
                print(f"   No improvement for {self.config.early_stopping_patience} epochs")
                break

        # Training complete
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"  Total time: {hours}h {minutes}m {seconds}s")
        print(f"  Best test accuracy: {self.best_test_acc:.2f}%")
        print(f"  Final learning rate: {current_lr:.6f}")
        print("="*70 + "\n")

        return self.history


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Important for:
        - Reproducible results
        - Debugging
        - Comparing different configurations
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    """
    Main training function.

    Usage:
        python train.py
        python train.py --epochs 500 --batch_size 128
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train EEGNet")
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create configuration
    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )

    # Create data loaders
    print("📊 Loading data...")
    train_loader, test_loader = create_data_loaders(
        data_path=str(config.data_dir),
        batch_size=config.batch_size,
        augment_train=config.augment_train,
        verbose=config.verbose
    )

    # Create model
    print("\n🧠 Creating model...")
    model = EEGNet(
        n_classes=config.n_classes,
        n_channels=config.n_channels,
        n_samples=config.n_samples,
        dropout_rate=config.dropout_rate,
        kernel_length=config.kernel_length,
        F1=config.F1,
        D=config.D,
        F2=config.F2,
        norm_rate=config.norm_rate,
        verbose=config.verbose
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config
    )

    # Train
    history = trainer.train()

    # Save final history
    history_path = config.save_dir / "training_history.npy"
    np.save(history_path, history)
    print(f"📊 Training history saved to: {history_path}")

    print("\n✨ Next steps:")
    print("  1. Evaluate model: python src/training/evaluate.py")
    print("  2. Visualize results: jupyter lab notebooks/03_model_training.ipynb")


if __name__ == "__main__":
    main()
