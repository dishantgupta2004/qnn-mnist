"""Utility functions for training, evaluation, and visualization."""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from sklearn.metrics import confusion_matrix
import seaborn as sns


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model predictions (logits or probabilities)
        labels: Ground truth labels
        
    Returns:
        Accuracy as percentage
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    n_samples: int = 10,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot sample images with true and predicted labels.
    
    Args:
        images: Image tensors (B, 1, 28, 28)
        true_labels: True labels
        pred_labels: Predicted labels
        n_samples: Number of samples to display
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(n_samples):
        img = images[i].squeeze().cpu().numpy()
        true_label = true_labels[i].item()
        pred_label = pred_labels[i].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(
            f'True: {true_label}, Pred: {pred_label}',
            color=color,
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for the output
    """
    print(f"\n{prefix}")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print("-" * 50)


