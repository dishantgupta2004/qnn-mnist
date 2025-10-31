"""Command-line script for training QNN model."""

import sys
from pathlib import Path
import argparse
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import load_config, save_config
from src.data import get_data_loaders
from src.qnn_model import create_model
from src.train import train_model
from src.eval import evaluate_model, print_evaluation_summary
from src.utils import seed_everything, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train Quantum Neural Network on MNIST")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--n-qubits", type=int, help="Override number of qubits")
    parser.add_argument("--layers", type=int, help="Override number of layers")
    parser.add_argument("--pca-k", type=int, help="Override PCA components")
    parser.add_argument("--train-size", type=int, help="Override training size")
    parser.add_argument("--val-size", type=int, help="Override validation size")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Override device")
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation on test set after training"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training plots"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from {args.config}...")
    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.n_qubits is not None:
        overrides['n_qubits'] = args.n_qubits
    if args.layers is not None:
        overrides['layers'] = args.layers
    if args.pca_k is not None:
        overrides['pca_k'] = args.pca_k
    if args.train_size is not None:
        overrides['train_size'] = args.train_size
    if args.val_size is not None:
        overrides['val_size'] = args.val_size
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.device is not None:
        overrides['device'] = args.device
    
    config = load_config(args.config, **overrides)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Set seed
    seed_everything(config.seed)
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        data_dir=config.data_dir,
        train_size=config.train_size,
        val_size=config.val_size,
        batch_size=config.batch_size,
        pca_k=config.pca_k,
        seed=config.seed
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, input_dim=input_dim)
    model.describe()
    
    # Create checkpoint path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"qnn_model_{timestamp}.pt"
    )
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device,
        checkpoint_path=checkpoint_path
    )
    
    # Save config alongside checkpoint
    config_path = checkpoint_path.replace('.pt', '_config.yaml')
    save_config(config, config_path)
    print(f"\n✓ Configuration saved to: {config_path}")
    
    # Save plots if requested
    if args.save_plots:
        plot_path = checkpoint_path.replace('.pt', '_training.png')
        plot_training_curves(history, save_path=plot_path, show=False)
        print(f"✓ Training plots saved to: {plot_path}")
    
    # Evaluate on test set
    if not args.no_eval:
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        
        # Load best checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate_model(model, test_loader, device=config.device)
        print_evaluation_summary(results)
        
        print(f"\nTest Accuracy: {results['accuracy']:.2f}%")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model checkpoint: {checkpoint_path}")
    print("="*60)


if __name__ == "__main__":
    main()