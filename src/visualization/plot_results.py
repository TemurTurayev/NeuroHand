"""
Training Results Visualization
================================

Функции для визуализации результатов обучения.

Автор: Temur Turayev
TashPMI, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional


def plot_training_history(
    history: Dict,
    figsize: tuple = (15, 5)
):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Training history dictionary
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    normalize: bool = False,
    figsize: tuple = (10, 8),
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Normalize to percentages
        figsize: Figure size
        title: Plot title
    """
    if class_names is None:
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        cbar_label = 'Percentage (%)'
    else:
        fmt = 'd'
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': cbar_label}
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_class_performance(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    class_names: list = None,
    figsize: tuple = (12, 6)
):
    """
    Plot per-class performance metrics.

    Args:
        precision: Precision per class
        recall: Recall per class
        f1: F1-score per class
        class_names: List of class names
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    return fig
