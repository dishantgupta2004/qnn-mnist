"""Evaluation utilities for Quantum Neural Network."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_outputs = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output.data, 1)
            
            # Accumulate
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_outputs.append(output.cpu())
            
            # Count correct
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calculate metrics
    accuracy = 100.0 * correct / total
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[str(i) for i in range(10)],
        output_dict=True
    )
    
    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "outputs": all_outputs,
        "confusion_matrix": cm,
        "classification_report": report
    }


def predict_single(
    model: nn.Module,
    image: torch.Tensor,
    device: str = "cpu"
) -> Tuple[int, torch.Tensor]:
    """
    Predict class for a single image.
    
    Args:
        model: Neural network model
        image: Input image tensor (1, 28, 28) or (28, 28) or (input_dim,)
        device: Device to predict on
        
    Returns:
        Tuple of (predicted_class, probabilities)
    """
    model.eval()
    model = model.to(device)
    
    # Ensure correct shape
    if image.dim() == 2:  # (28, 28)
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    elif image.dim() == 3:  # (1, 28, 28)
        image = image.unsqueeze(0)  # (1, 1, 28, 28)
    elif image.dim() == 1:  # (input_dim,)
        image = image.unsqueeze(0)  # (1, input_dim)
    
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities[0]


def get_model_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions for a dataset.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Device to predict on
        max_samples: Maximum number of samples to predict (None for all)
        
    Returns:
        Tuple of (images, labels, predictions)
    """
    model.eval()
    model = model.to(device)
    
    all_images = []
    all_labels = []
    all_predictions = []
    
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            if max_samples and total_samples >= max_samples:
                break
            
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            # Store
            all_images.append(data.cpu())
            all_labels.append(target.cpu())
            all_predictions.append(predicted.cpu())
            
            total_samples += data.size(0)
    
    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Limit to max_samples if specified
    if max_samples:
        all_images = all_images[:max_samples]
        all_labels = all_labels[:max_samples]
        all_predictions = all_predictions[:max_samples]
    
    return all_images.numpy(), all_labels.numpy(), all_predictions.numpy()


def print_evaluation_summary(results: Dict):
    """
    Print evaluation summary.
    
    Args:
        results: Results dictionary from evaluate_model
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    report = results['classification_report']
    for i in range(10):
        class_name = str(i)
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<10} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f}")
    
    print("-" * 60)
    
    # Macro average
    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"{'Macro Avg':<10} "
              f"{macro['precision']:<12.4f} "
              f"{macro['recall']:<12.4f} "
              f"{macro['f1-score']:<12.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test evaluation
    from src.config import load_config
    from src.data import get_data_loaders
    from src.qnn_model import create_model
    from src.utils import seed_everything
    
    # Load config
    config = load_config("configs/default.yaml")
    config.train_size = 500
    config.val_size = 100
    
    # Set seed
    seed_everything(config.seed)
    
    # Prepare data
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        data_dir=config.data_dir,
        train_size=config.train_size,
        val_size=config.val_size,
        batch_size=config.batch_size,
        pca_k=config.pca_k,
        seed=config.seed
    )
    
    # Create model (randomly initialized)
    model = create_model(config, input_dim=input_dim)
    
    print("Evaluating randomly initialized model...")
    results = evaluate_model(model, test_loader, device=config.device)
    
    print_evaluation_summary(results)
    
    # Test single prediction
    print("\nTesting single prediction:")
    data, label = next(iter(test_loader))
    image = data[0]
    
    pred_class, probabilities = predict_single(model, image, device=config.device)
    print(f"True label: {label[0].item()}")
    print(f"Predicted class: {pred_class}")
    print(f"Top 3 probabilities:")
    top3_probs, top3_classes = torch.topk(probabilities, 3)
    for i in range(3):
        print(f"  Class {top3_classes[i].item()}: {top3_probs[i].item():.4f}")