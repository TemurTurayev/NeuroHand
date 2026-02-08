"""
Model Evaluation
=================

Evaluation и метрики для trained EEGNet model.

Автор: Temur Turayev
TashPMI, 2024
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)
from tqdm import tqdm

from src.constants import CLASS_NAMES, N_CLASSES, N_CHANNELS, N_SAMPLES
from src.models.eegnet import EEGNet
from src.models.utils import load_checkpoint
from src.data.dataset import create_data_loaders

PROJECT_ROOT = Path(__file__).parent.parent.parent


def calculate_itr(n_classes: int, accuracy: float, trial_duration: float = 4.0) -> float:
    """Calculate Information Transfer Rate in bits/min."""
    if accuracy <= 0 or accuracy >= 1:
        return 0.0
    N = n_classes
    P = accuracy
    itr_per_trial = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    trials_per_min = 60.0 / trial_duration
    return itr_per_trial * trials_per_min


class ModelEvaluator:
    """
    Evaluator для trained EEGNet model.

    Computes:
        - Accuracy
        - Precision, Recall, F1-score
        - Confusion matrix
        - Per-class metrics
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = 'cpu',
        class_names: list = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained EEGNet model
            test_loader: Test data loader
            device: Device to use
            class_names: List of class names for reporting
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

        if class_names is None:
            self.class_names = list(CLASS_NAMES)
        else:
            self.class_names = class_names

        self.model.eval()

    @torch.no_grad()
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for entire test set.

        Returns:
            y_true: True labels
            y_pred: Predicted labels
        """
        all_labels = []
        all_predictions = []

        print("🔮 Generating predictions...")
        for signals, labels in tqdm(self.test_loader):
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(signals)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        return np.array(all_labels), np.array(all_predictions)

    def evaluate(self) -> Dict:
        """
        Comprehensive evaluation.

        Returns:
            Dictionary with all metrics
        """
        # Get predictions
        y_true, y_pred = self.predict()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        cm = confusion_matrix(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        itr = calculate_itr(len(self.class_names), accuracy)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'confusion_matrix': cm,
            'kappa': kappa,
            'itr': itr,
            'y_true': y_true,
            'y_pred': y_pred
        }

        return results

    def print_results(self, results: Dict):
        """
        Print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)

        print(f"\n🎯 Overall Accuracy: {results['accuracy']*100:.2f}%")
        print(f"   Cohen's Kappa:   {results['kappa']:.4f}")
        print(f"   ITR:             {results['itr']:.2f} bits/min\n")

        print("Per-Class Metrics:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)

        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15} "
                  f"{results['precision'][i]:<12.4f} "
                  f"{results['recall'][i]:<12.4f} "
                  f"{results['f1'][i]:<12.4f} "
                  f"{results['support'][i]:<10}")

        # Average metrics
        avg_precision = results['precision'].mean()
        avg_recall = results['recall'].mean()
        avg_f1 = results['f1'].mean()

        print("-" * 70)
        print(f"{'Average':<15} "
              f"{avg_precision:<12.4f} "
              f"{avg_recall:<12.4f} "
              f"{avg_f1:<12.4f}")
        print("-" * 70)

        # Confusion matrix
        print("\n📈 Confusion Matrix:")
        print("-" * 70)
        print(f"{'':>15}", end='')
        for class_name in self.class_names:
            print(f"{class_name[:10]:>12}", end='')
        print()

        cm = results['confusion_matrix']
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15}", end='')
            for j in range(len(self.class_names)):
                print(f"{cm[i, j]:>12}", end='')
            print()

        print("="*70 + "\n")

        # Medical interpretation
        print("🏥 MEDICAL INTERPRETATION:")
        print("-" * 70)
        if results['accuracy'] >= 0.80:
            print("✅ Excellent performance (≥80%)")
            print("   Model is ready for real-world testing")
        elif results['accuracy'] >= 0.70:
            print("✅ Good performance (70-80%)")
            print("   Model shows promise, consider more training data")
        elif results['accuracy'] >= 0.60:
            print("⚠️  Moderate performance (60-70%)")
            print("   May need hyperparameter tuning or more data")
        else:
            print("❌ Poor performance (<60%)")
            print("   Consider data quality, preprocessing, or model architecture")

        print("\nPer-Class Analysis:")
        for i, class_name in enumerate(self.class_names):
            if results['f1'][i] < 0.60:
                print(f"  ⚠️  {class_name}: Low F1-score ({results['f1'][i]:.2f})")
                print(f"     → May need more training data for this class")

        print("="*70 + "\n")


def main():
    """
    Main evaluation function.

    Usage:
        python evaluate.py
        python evaluate.py --checkpoint models/checkpoints/best_model.pth
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained EEGNet")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Path to processed data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto/cuda/mps/cpu)'
    )

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"📱 Using device: {device}")

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train model first: python src/training/train.py")
        return

    # Load data
    print(f"\n📊 Loading data from: {args.data_dir}")
    _, test_loader = create_data_loaders(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        verbose=True
    )

    # Create model
    print("\n🧠 Loading model...")
    model = EEGNet(
        n_classes=N_CLASSES,
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        verbose=False
    )

    # Load checkpoint
    checkpoint = load_checkpoint(
        filepath=str(checkpoint_path),
        model=model,
        device=device
    )

    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Training accuracy: {checkpoint.get('accuracy', 'N/A')}")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device
    )

    # Evaluate
    results = evaluator.evaluate()

    # Print results
    evaluator.print_results(results)

    # Save results
    save_dir = Path("models/checkpoints")
    np.save(save_dir / "evaluation_results.npy", results)
    print(f"💾 Results saved to: {save_dir / 'evaluation_results.npy'}")


if __name__ == "__main__":
    main()
