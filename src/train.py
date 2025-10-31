"""Training loop for Quantum Neural Network."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time
from src.utils import calculate_accuracy


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        running_correct += (predicted == target).sum().item()
        total_samples += data.size(0)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            batch_acc = 100.0 * (predicted == target).sum().item() / data.size(0)
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss = {loss.item():.4f}, Acc = {batch_acc:.2f}%")
    
    avg_loss = running_loss / total_samples
    avg_acc = 100.0 * running_correct / total_samples
    
    return {
        "loss": avg_loss,
        "accuracy": avg_acc
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Track metrics
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            running_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
    
    avg_loss = running_loss / total_samples
    avg_acc = 100.0 * running_correct / total_samples
    
    return {
        "loss": avg_loss,
        "accuracy": avg_acc
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Complete training loop.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        device: Device to train on
        checkpoint_path: Path to save best checkpoint
        
    Returns:
        Training history dictionary
    """
    # Move model to device
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    total_start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            
            if checkpoint_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'config': config.to_dict()
                }, checkpoint_path)
                print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print("="*60 + "\n")
    
    return history


if __name__ == "__main__":
    # Test training loop
    from src.config import load_config
    from src.data import get_data_loaders
    from src.qnn_model import create_model
    from src.utils import seed_everything
    
    # Load config
    config = load_config("configs/default.yaml")
    config.epochs = 2
    config.train_size = 1000
    config.val_size = 200
    
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
    
    # Create model
    model = create_model(config, input_dim=input_dim)
    model.describe()
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device,
        checkpoint_path="test_checkpoint.pt"
    )
    
    print("\nTraining history:")
    print(history)