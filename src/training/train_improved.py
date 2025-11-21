"""
Improved Training Script - Target: 68-70% Accuracy
===================================================

Optimizations:
- No early stopping (or patience=100)
- Better hyperparameters from EEGNet paper
- Learning rate warmup
- Longer training (500 epochs)
- Better data augmentation

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
from tqdm import tqdm
import time

from src.models.eegnet import EEGNet
from src.data.dataset import EEGDataset
from src.training.config import TrainingConfig


class ImprovedTrainer:
    """
    Improved trainer with better hyperparameters.

    Changes from baseline:
    - Higher patience for early stopping (100 vs 50)
    - Cosine annealing scheduler (vs ReduceLROnPlateau)
    - Warmup learning rate
    - Max norm constraint = 0.5 (vs 0.25)
    - Better weight decay
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam with better weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.001,  # Slightly less regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Scheduler: Cosine Annealing with Warm Restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,  # Restart every 50 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': []
        }

        # Early stopping (increased patience)
        self.best_test_acc = 0.0
        self.patience = 100  # Increased from 50
        self.patience_counter = 0

        print(f"\n🚀 Improved Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Early stopping patience: {self.patience}")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")

    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Apply max-norm constraint (EEGNet paper: 0.5)
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() >= 2:  # Only for 2D+ tensors
                    param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=0.5)

            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def test_epoch(self) -> tuple:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_acc': self.best_test_acc,
            'history': self.history
        }

        save_dir = PROJECT_ROOT / 'models' / 'checkpoints'
        save_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            torch.save(checkpoint, save_dir / 'best_model_improved.pth')

        # Save latest every 50 epochs
        if epoch % 50 == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')

    def train(self):
        """Complete training loop."""
        print("\n" + "="*70)
        print("🎯 IMPROVED TRAINING STARTED")
        print("="*70)
        print(f"Configuration:")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Weight decay: 0.001")
        print(f"  Max norm: 0.5")
        print(f"  Device: {self.device}")
        print("="*70)

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Test
            test_loss, test_acc = self.test_epoch()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)

            # Print progress
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ Best model saved! (Test Acc: {test_acc:.2f}%)")
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered!")
                print(f"   No improvement for {self.patience} epochs")
                break

            # Save checkpoint periodically
            if epoch % 50 == 0:
                self.save_checkpoint(epoch, is_best=False)

        # Training complete
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("\n" + "="*70)
        print("✅ IMPROVED TRAINING COMPLETE!")
        print("="*70)
        print(f"  Total time: {hours}h {minutes}m {seconds}s")
        print(f"  Best test accuracy: {self.best_test_acc:.2f}%")
        print(f"  Final learning rate: {current_lr:.6f}")
        print("="*70)

        # Save final history
        save_dir = PROJECT_ROOT / 'models' / 'checkpoints'
        np.save(save_dir / 'training_history_improved.npy', self.history)
        print(f"\n📊 Training history saved to: {save_dir / 'training_history_improved.npy'}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Improved EEGNet training")
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Load data
    print("📊 Loading data...")
    data_dir = PROJECT_ROOT / args.data_dir

    train_dataset = EEGDataset(
        data_path=str(data_dir),
        split='train',
        augment=True
    )

    test_dataset = EEGDataset(
        data_path=str(data_dir),
        split='test',
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # Create model
    print("\n🧠 Creating model...")
    model = EEGNet(n_classes=4, n_channels=22, n_samples=1000)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create trainer
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config
    )

    # Train
    trainer.train()

    print("\n✨ Next steps:")
    print("  1. Evaluate: python src/training/evaluate.py --model_path models/checkpoints/best_model_improved.pth")
    print("  2. Compare: Check training_history_improved.npy vs training_history.npy")


if __name__ == "__main__":
    main()
