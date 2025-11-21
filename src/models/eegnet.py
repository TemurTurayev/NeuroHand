"""
EEGNet: Compact Convolutional Neural Network for EEG-based BCIs
================================================================

Реализация EEGNet архитектуры from scratch с подробными объяснениями.

Reference Paper:
    Lawhern, V. J., et al. (2018).
    "EEGNet: A compact convolutional neural network for EEG-based
    brain–computer interfaces."
    Journal of Neural Engineering, 15(5), 056013.

Почему EEGNet?
    - Compact: ~3000 parameters (vs millions in typical CNNs)
    - Fast: ~10ms inference time
    - Effective: State-of-the-art on motor imagery tasks
    - Interpretable: Learns spatial filters similar to CSP
    - Generalizable: Works across different BCI paradigms

Автор: Temur Turayev
TashPMI, 2024
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class EEGNet(nn.Module):
    """
    EEGNet architecture for EEG classification.

    Architecture Overview:
        Input: [batch, 1, channels, samples]
              ↓
        Block 1: Temporal + Spatial Filtering
              ↓
        Block 2: Separable Convolution
              ↓
        Classifier: Fully Connected Layer
              ↓
        Output: [batch, n_classes]

    Medical Interpretation:
        Block 1: Learns "what" brain patterns (similar to EEG montages)
        Block 2: Learns "when" these patterns occur (temporal features)
        Result: Identifies motor imagery patterns in brain signals
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_samples: int = 1000,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
        verbose: bool = False
    ):
        """
        Initialize EEGNet model.

        Args:
            n_classes: Number of output classes (4 for motor imagery)
            n_channels: Number of EEG channels (22 for BCI IV-2a, 8 for OpenBCI)
            n_samples: Number of time samples (1000 for 4s at 250Hz)
            dropout_rate: Dropout probability (0.5 = 50% dropout)
            kernel_length: Length of temporal convolution (64 = 256ms at 250Hz)
            F1: Number of temporal filters (default: 8)
            D: Depth multiplier for spatial filters (default: 2)
            F2: Number of pointwise filters (default: 16, usually F1*D)
            norm_rate: Max norm constraint for dense layer
            verbose: Print model architecture

        Hyperparameter Explanation:
            - F1: How many different temporal patterns to learn
            - D: How many spatial patterns per temporal pattern
            - F2: How many combined features to extract
            - kernel_length: How much time context to use (longer = more context)
        """
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate

        # =================================================================
        # BLOCK 1: Temporal Convolution + Spatial Filtering
        # =================================================================

        # Layer 1.1: Temporal Convolution
        # ---------------------------------
        # Purpose: Learn temporal filters (frequency patterns over time)
        # Input: [batch, 1, channels, samples]
        # Output: [batch, F1, channels, samples//2]
        #
        # Medical Interpretation:
        #   - Learns to detect oscillatory patterns (alpha, beta, gamma rhythms)
        #   - Each filter learns a different temporal pattern
        #   - Kernel size (1, kernel_length) means: process each channel independently over time
        self.conv1 = nn.Conv2d(
            in_channels=1,          # Input: single time series
            out_channels=F1,        # Output: F1 different temporal filters
            kernel_size=(1, kernel_length),  # (spatial, temporal)
            padding=(0, kernel_length // 2),  # Same padding for temporal dimension
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Layer 1.2: Depthwise Spatial Convolution
        # ------------------------------------------
        # Purpose: Learn spatial filters (how channels relate to each other)
        # Output: [batch, F1*D, 1, samples//2]
        #
        # Medical Interpretation:
        #   - Learns spatial patterns similar to Common Spatial Patterns (CSP)
        #   - Identifies which brain regions are relevant for each temporal pattern
        #   - Depthwise = each temporal filter gets D independent spatial filters
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,    # Depth multiplier
            kernel_size=(n_channels, 1),  # Process all channels at once
            groups=F1,              # Depthwise: each input channel processed separately
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()  # ELU activation (smoother than ReLU)
        self.pool1 = nn.AvgPool2d((1, 4))  # Downsample time by 4
        self.dropout1 = nn.Dropout(dropout_rate)

        # =================================================================
        # BLOCK 2: Separable Convolution
        # =================================================================

        # Layer 2.1: Depthwise Temporal Convolution
        # -------------------------------------------
        # Purpose: Learn more temporal patterns within each spatial filter
        # Output: [batch, F1*D, 1, samples//8]
        self.separable_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),    # Temporal kernel
            padding=(0, 8),         # Same padding
            groups=F1 * D,          # Depthwise
            bias=False
        )

        # Layer 2.2: Pointwise Convolution
        # ----------------------------------
        # Purpose: Combine features from different spatial filters
        # Output: [batch, F2, 1, samples//8]
        self.pointwise_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),     # 1x1 convolution
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))  # Downsample time by 8
        self.dropout2 = nn.Dropout(dropout_rate)

        # =================================================================
        # CLASSIFIER: Fully Connected Layer
        # =================================================================

        # Calculate size after all pooling operations
        # Initial: n_samples
        # After pool1: n_samples // 4
        # After pool2: n_samples // 4 // 8 = n_samples // 32
        self.flatten_size = F2 * (n_samples // 32)

        # Dense layer with max norm constraint
        # This prevents overfitting by limiting weight magnitudes
        self.fc = nn.Linear(self.flatten_size, n_classes)

        # Store max norm constraint (applied during training)
        self.fc.weight.data.normal_(0, 0.01)  # Initialize with small values
        self.fc.bias.data.zero_()

        if verbose:
            self.print_architecture()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEGNet.

        Args:
            x: Input EEG data [batch, 1, channels, samples]

        Returns:
            Class logits [batch, n_classes]

        Forward Pass Explanation:
            1. Input: Raw EEG signals
            2. Block 1: Extract spatial-temporal patterns
            3. Block 2: Refine and combine patterns
            4. Classifier: Map to class probabilities
        """
        # =================================================================
        # BLOCK 1
        # =================================================================

        # Temporal convolution
        x = self.conv1(x)  # [batch, F1, channels, samples]
        x = self.batchnorm1(x)

        # Spatial filtering (depthwise)
        x = self.depthwise_conv(x)  # [batch, F1*D, 1, samples]
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pool1(x)  # [batch, F1*D, 1, samples//4]
        x = self.dropout1(x)

        # =================================================================
        # BLOCK 2
        # =================================================================

        # Separable convolution (depthwise + pointwise)
        x = self.separable_conv(x)  # [batch, F1*D, 1, samples//4]
        x = self.pointwise_conv(x)  # [batch, F2, 1, samples//4]
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pool2(x)  # [batch, F2, 1, samples//32]
        x = self.dropout2(x)

        # =================================================================
        # CLASSIFIER
        # =================================================================

        # Flatten
        x = x.view(x.size(0), -1)  # [batch, F2 * (samples//32)]

        # Dense layer
        x = self.fc(x)  # [batch, n_classes]

        return x

    def apply_max_norm_constraint(self):
        """
        Apply max norm constraint to final dense layer.

        This is a form of regularization that prevents weights from growing too large.
        Called after each optimizer step during training.

        Why Max Norm?
            - Prevents overfitting
            - Stabilizes training
            - Common in EEG classification (small datasets)
        """
        with torch.no_grad():
            norm = self.fc.weight.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, 0, self.norm_rate)
            self.fc.weight *= (desired / (1e-8 + norm))

    def print_architecture(self):
        """Print model architecture summary."""
        print("\n" + "="*70)
        print("🧠 EEGNet ARCHITECTURE")
        print("="*70)
        print(f"Input shape: [batch, 1, {self.n_channels}, {self.n_samples}]")
        print("\nBLOCK 1: Temporal + Spatial Filtering")
        print(f"  Conv2D (temporal):    [batch, {self.F1}, {self.n_channels}, {self.n_samples}]")
        print(f"  DepthwiseConv2D:      [batch, {self.F1*self.D}, 1, {self.n_samples}]")
        print(f"  AvgPool2D:            [batch, {self.F1*self.D}, 1, {self.n_samples//4}]")
        print("\nBLOCK 2: Separable Convolution")
        print(f"  SeparableConv2D:      [batch, {self.F2}, 1, {self.n_samples//32}]")
        print("\nCLASSIFIER")
        print(f"  Flatten:              [batch, {self.flatten_size}]")
        print(f"  Dense:                [batch, {self.n_classes}]")
        print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("="*70 + "\n")

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    n_classes: int = 4,
    n_channels: int = 22,
    n_samples: int = 1000,
    device: Optional[str] = None,
    verbose: bool = True
) -> EEGNet:
    """
    Create and initialize EEGNet model.

    Args:
        n_classes: Number of classes
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        device: Device to use ('cuda', 'mps', or 'cpu')
        verbose: Print model information

    Returns:
        Initialized EEGNet model

    Usage:
        >>> model = create_model(n_classes=4, n_channels=22, device='mps')
        >>> x = torch.randn(32, 1, 22, 1000)  # Batch of 32 samples
        >>> y = model(x)  # [32, 4]
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
        else:
            device = 'cpu'

    # Create model
    model = EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        verbose=verbose
    )

    # Move to device
    model = model.to(device)

    if verbose:
        print(f"✅ Model created and moved to: {device}")
        if device == 'mps':
            print("   (Using Apple Silicon GPU acceleration! 🚀)")

    return model


def main():
    """
    Test EEGNet model creation and forward pass.

    Usage:
        python eegnet.py
    """
    print("="*70)
    print("🧪 TESTING EEGNET MODEL")
    print("="*70)

    # Create model
    model = create_model(
        n_classes=4,
        n_channels=22,
        n_samples=1000,
        verbose=True
    )

    # Create dummy input
    batch_size = 16
    x = torch.randn(batch_size, 1, 22, 1000)

    print(f"\n🔍 Testing forward pass...")
    print(f"   Input shape: {x.shape}")

    # Forward pass
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        output = model(x)

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test softmax (convert to probabilities)
    probs = F.softmax(output, dim=1)
    print(f"\n📊 Class probabilities (first sample):")
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    for i, prob in enumerate(probs[0]):
        print(f"   {class_names[i]}: {prob:.4f}")

    # Test max norm constraint
    print(f"\n🔧 Testing max norm constraint...")
    initial_norm = model.fc.weight.norm(2, dim=1).mean()
    print(f"   Initial weight norm: {initial_norm:.4f}")

    model.apply_max_norm_constraint()

    final_norm = model.fc.weight.norm(2, dim=1).mean()
    print(f"   After constraint: {final_norm:.4f}")

    print("\n✅ EEGNet test passed!")
    print("="*70)


if __name__ == "__main__":
    main()
