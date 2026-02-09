"""
Real-time Inference for EEGNet
===============================

Make predictions on new EEG data using the trained model.

Usage:
    python src/inference/predict.py --input path/to/eeg_data.npy
    python src/inference/predict.py --demo  # Use random test sample

Автор: Temur Turayev
TashPMI, 2024
"""

from pathlib import Path

# Project root for file resolution (not added to sys.path)
PROJECT_ROOT = Path(__file__).parent.parent.parent

import argparse
import numpy as np
import torch
import time

from src.constants import CLASS_NAMES, N_CLASSES, N_CHANNELS, N_SAMPLES
from src.models.eegnet import EEGNet
from src.models.utils import get_device
from src.data.dataset import EEGDataset
from src.logging_config import get_logger

logger = get_logger(__name__)


class EEGPredictor:
    """
    Inference class for making predictions with trained EEGNet.

    Example:
        >>> predictor = EEGPredictor('models/checkpoints/best_model.pth')
        >>> signal = np.random.randn(22, 1000)  # [channels, samples]
        >>> prediction = predictor.predict(signal)
        >>> print(f"Predicted class: {prediction['class_name']}")
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'auto'
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            device: 'auto', 'mps', 'cuda', or 'cpu'
        """
        # Set device
        self.device = get_device(device)

        logger.info("Using device: %s", self.device)

        # Load model
        self.model = EEGNet(
            n_classes=N_CLASSES,
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
        ).to(self.device)
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Class names
        self.class_names = list(CLASS_NAMES)

        logger.info("Model loaded from: %s (trained for %d epochs)", model_path, checkpoint['epoch'])
        if 'train_acc' in checkpoint:
            logger.info("Training accuracy: %.2f%%", checkpoint['train_acc'])

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess EEG signal for inference.

        Args:
            signal: EEG data [channels, samples] or [batch, channels, samples]

        Returns:
            Preprocessed tensor ready for model
        """
        # Add batch dimension if needed
        if signal.ndim == 2:
            signal = signal[np.newaxis, ...]  # [1, channels, samples]

        # Convert to tensor
        signal = torch.from_numpy(signal).float()

        # Add channel dimension: [batch, 1, channels, samples]
        if signal.ndim == 3:
            signal = signal.unsqueeze(1)

        return signal.to(self.device)

    def predict(
        self,
        signal: np.ndarray,
        return_probs: bool = True
    ) -> dict:
        """
        Make prediction on EEG signal.

        Args:
            signal: EEG data [channels, samples]
            return_probs: Return class probabilities

        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        x = self.preprocess(signal)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        inference_time = (time.time() - start_time) * 1000  # milliseconds

        # Format results
        result = {
            'class_id': pred_class,
            'class_name': self.class_names[pred_class],
            'confidence': confidence,
            'inference_time_ms': inference_time
        }

        if return_probs:
            result['probabilities'] = {
                name: prob.item()
                for name, prob in zip(self.class_names, probs[0])
            }

        return result

    def predict_batch(
        self,
        signals: np.ndarray
    ) -> list:
        """
        Make predictions on batch of signals.

        Args:
            signals: Batch of EEG data [batch, channels, samples]

        Returns:
            List of prediction dictionaries
        """
        x = self.preprocess(signals)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)

        results = []
        for i in range(len(signals)):
            result = {
                'class_id': pred_classes[i].item(),
                'class_name': self.class_names[pred_classes[i].item()],
                'confidence': probs[i, pred_classes[i]].item(),
                'probabilities': {
                    name: probs[i, j].item()
                    for j, name in enumerate(self.class_names)
                }
            }
            results.append(result)

        return results


def demo_prediction(predictor: EEGPredictor, data_dir: str = 'data/processed'):
    """
    Demo prediction using random test sample.

    Args:
        predictor: Initialized EEGPredictor
        data_dir: Directory with processed data
    """
    logger.info("DEMO PREDICTION")

    # Load test data
    data_dir = Path(data_dir)
    X_test = np.load(data_dir / 'test_data.npy')
    y_test = np.load(data_dir / 'test_labels.npy')

    # Random sample
    idx = np.random.randint(0, len(X_test))
    signal = X_test[idx]
    true_label = y_test[idx]
    true_class = predictor.class_names[true_label]

    logger.info("Sample %d: true_class=%s (ID=%d), shape=%s",
                idx, true_class, true_label, signal.shape)

    # Make prediction
    result = predictor.predict(signal)

    # Display results
    logger.info("Prediction: %s (ID=%d), confidence=%.2f%%, inference_time=%.2f ms",
                result['class_name'], result['class_id'],
                result['confidence'] * 100, result['inference_time_ms'])

    for name, prob in result['probabilities'].items():
        logger.info("  %s: %.2f%%", name, prob * 100)

    # Correctness
    is_correct = result['class_id'] == true_label
    if is_correct:
        logger.info("Prediction: CORRECT")
    else:
        logger.warning("Prediction: INCORRECT (predicted=%s, true=%s)",
                       result['class_name'], true_class)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained EEGNet model"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input EEG data (.npy file) [channels, samples]'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo prediction on random test sample'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='Device to use for inference'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = EEGPredictor(
        model_path=args.model_path,
        device=args.device
    )

    # Demo mode
    if args.demo:
        demo_prediction(predictor)
        return

    # Custom input
    if args.input:
        print(f"\n📂 Loading data from: {args.input}")
        signal = np.load(args.input)

        print(f"   Signal shape: {signal.shape}")

        # Single prediction
        if signal.ndim == 2:
            result = predictor.predict(signal)

            print(f"\n🔮 Prediction:")
            print(f"   Class: {result['class_name']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print(f"   Inference time: {result['inference_time_ms']:.2f} ms")

            print(f"\n📊 Probabilities:")
            for name, prob in result['probabilities'].items():
                print(f"   {name}: {prob*100:.2f}%")

        # Batch prediction
        else:
            print(f"\n🔮 Batch prediction on {len(signal)} samples...")
            results = predictor.predict_batch(signal)

            for i, result in enumerate(results):
                print(f"\n   Sample {i}: {result['class_name']} "
                      f"({result['confidence']*100:.2f}%)")

        return

    # No input specified
    print("\n⚠️  No input specified. Use --demo or --input <path>")
    print("Examples:")
    print("  python src/inference/predict.py --demo")
    print("  python src/inference/predict.py --input data.npy")


if __name__ == "__main__":
    main()
