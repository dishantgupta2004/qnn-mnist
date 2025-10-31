"""Command-line script for evaluating QNN model."""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import Config
from src.data import get_data_loaders
from src.qnn_model import create_model
from src.eval import evaluate_model, print_evaluation_summary
from src.utils import seed_everything, plot_confusion_matrix



def main():
    parser = argparse.ArgumentParser(description="Evaluate Quantum Neural Network on MNIST")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--save-confusion-matrix",
        action="store_true",
        help="Save confusion matrix plot"
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config.from_dict(config_dict)
        print("✓ Configuration loaded from checkpoint")
    else:
        print("Warning: No config found in checkpoint, using defaults")
        config = Config()
    
    # Override device and batch size
    config.device = args.device
    config.batch_size = args.batch_size
    
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Set seed
    seed_everything(config.seed)
    
    # Prepare data (test set only)
    print("\nLoading test data...")
    _, _, test_loader, input_dim = get_data_loaders(
        data_dir=args.data_dir,
        train_size=config.train_size,
        val_size=config.val_size,
        batch_size=config.batch_size,
        pca_k=config.pca_k,
        seed=config.seed
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, input_dim=input_dim)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model weights loaded")
    
    if 'epoch' in checkpoint:
        print(f"✓ Checkpoint from epoch {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"✓ Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    results = evaluate_model(model, test_loader, device=args.device)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save confusion matrix if requested
    if args.save_confusion_matrix:
        save_path = args.checkpoint.replace('.pt', '_confusion_matrix.png')
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            save_path=save_path,
            show=False
        )
        print(f"\n✓ Confusion matrix saved to: {save_path}")
    
    print("\n" + "="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()