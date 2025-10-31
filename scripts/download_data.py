"""Script to pre-download MNIST dataset."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets
import argparse


def download_mnist(data_dir: str = "data"):
    """
    Download MNIST dataset.
    
    Args:
        data_dir: Directory to save dataset
    """
    print(f"Downloading MNIST dataset to {data_dir}...")
    
    # Download training set
    print("\nDownloading training set...")
    datasets.MNIST(
        root=data_dir,
        train=True,
        download=True
    )
    
    # Download test set
    print("\nDownloading test set...")
    datasets.MNIST(
        root=data_dir,
        train=False,
        download=True
    )
    
    print("\n✓ MNIST dataset downloaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save dataset (default: data)"
    )
    
    args = parser.parse_args()
    
    download_mnist(args.data_dir)