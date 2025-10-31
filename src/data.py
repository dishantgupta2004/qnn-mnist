"""MNIST data loading and preprocessing."""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os


class MNISTDataModule:
    """Data module for MNIST with optional PCA preprocessing."""
    
    def __init__(
        self,
        data_dir: str = "data",
        train_size: int = 10000,
        val_size: int = 2000,
        batch_size: int = 64,
        pca_k: int = 0,
        seed: int = 42
    ):
        """
        Initialize MNIST data module.
        
        Args:
            data_dir: Directory to store MNIST data
            train_size: Number of training samples to use
            val_size: Number of validation samples to use
            batch_size: Batch size for data loaders
            pca_k: Number of PCA components (0 to disable)
            seed: Random seed
        """
        self.data_dir = data_dir
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.pca_k = pca_k
        self.seed = seed
        
        self.pca = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_dim = 784 if pca_k == 0 else pca_k
        
    def prepare_data(self):
        """Download MNIST dataset if not already present."""
        datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True
        )
        datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True
        )
        
    def setup(self):
        """Setup train/val/test splits and apply PCA if requested."""
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Load full datasets
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transform,
            download=False
        )
        
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transform,
            download=False
        )
        
        # Create train/val split
        torch.manual_seed(self.seed)
        total_train = min(self.train_size + self.val_size, len(train_dataset))
        indices = torch.randperm(len(train_dataset))[:total_train].tolist()
        
        train_indices = indices[:self.train_size]
        val_indices = indices[self.train_size:self.train_size + self.val_size]
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        # Apply PCA if requested
        if self.pca_k > 0:
            print(f"Applying PCA: 784 -> {self.pca_k} components")
            train_subset = self._apply_pca_transform(train_subset, fit=True)
            val_subset = self._apply_pca_transform(val_subset, fit=False)
            test_dataset = self._apply_pca_transform(test_dataset, fit=False)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Data prepared:")
        print(f"  Training samples: {len(train_subset)}")
        print(f"  Validation samples: {len(val_subset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Input dimension: {self.input_dim}")
        
    def _apply_pca_transform(self, dataset, fit: bool = False):
        """Apply PCA transformation to dataset."""
        # Extract all images and labels
        images = []
        labels = []
        
        for img, label in dataset:
            images.append(img.flatten().numpy())
            labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Fit PCA on training data
        if fit:
            self.scaler = StandardScaler()
            images_scaled = self.scaler.fit_transform(images)
            
            self.pca = PCA(n_components=self.pca_k, random_state=self.seed)
            images_pca = self.pca.fit_transform(images_scaled)
            
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(f"  PCA explained variance: {explained_var:.4f}")
        else:
            images_scaled = self.scaler.transform(images)
            images_pca = self.pca.transform(images_scaled)
        
        # Create new dataset
        return PCADataset(images_pca, labels)


class PCADataset(torch.utils.data.Dataset):
    """Dataset wrapper for PCA-transformed data."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """
        Initialize PCA dataset.
        
        Args:
            images: PCA-transformed images (N, k)
            labels: Labels (N,)
        """
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def get_data_loaders(
    data_dir: str = "data",
    train_size: int = 10000,
    val_size: int = 2000,
    batch_size: int = 64,
    pca_k: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Convenience function to get MNIST data loaders.
    
    Args:
        data_dir: Directory to store MNIST data
        train_size: Number of training samples
        val_size: Number of validation samples
        batch_size: Batch size
        pca_k: Number of PCA components (0 to disable)
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_dim)
    """
    data_module = MNISTDataModule(
        data_dir=data_dir,
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
        pca_k=pca_k,
        seed=seed
    )
    
    data_module.prepare_data()
    data_module.setup()
    
    return (
        data_module.train_loader,
        data_module.val_loader,
        data_module.test_loader,
        data_module.input_dim
    )


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading without PCA:")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        train_size=1000,
        val_size=200,
        batch_size=32,
        pca_k=0
    )
    
    # Check batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Input dimension: {input_dim}")
    
    print("\n" + "="*50 + "\n")
    
    print("Testing data loading with PCA:")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        train_size=1000,
        val_size=200,
        batch_size=32,
        pca_k=8
    )
    
    # Check batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Input dimension: {input_dim}")