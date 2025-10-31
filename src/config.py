"""Configuration loading and management."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration dataclass for QNN-MNIST."""
    
    # Random seed
    seed: int = 42
    
    # Quantum circuit
    n_qubits: int = 8
    layers: int = 3
    
    # Data preprocessing
    pca_k: int = 8
    
    # Training
    epochs: int = 10
    batch_size: int = 64
    lr: float = 0.001
    
    # Dataset
    train_size: int = 10000
    val_size: int = 2000
    
    # Device
    device: str = "cpu"
    shots: Optional[int] = None
    
    # Paths
    checkpoint_dir: str = "data/checkpoints"
    data_dir: str = "data"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")
        if self.layers < 1:
            raise ValueError(f"layers must be >= 1, got {self.layers}")
        if self.pca_k > 0 and self.pca_k < self.n_qubits:
            raise ValueError(f"pca_k ({self.pca_k}) should be >= n_qubits ({self.n_qubits})")
        if self.train_size < 1 or self.train_size > 60000:
            raise ValueError(f"train_size must be in [1, 60000], got {self.train_size}")
        if self.val_size < 1:
            raise ValueError(f"val_size must be >= 1, got {self.val_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "seed": self.seed,
            "n_qubits": self.n_qubits,
            "layers": self.layers,
            "pca_k": self.pca_k,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "device": self.device,
            "shots": self.shots,
            "checkpoint_dir": self.checkpoint_dir,
            "data_dir": self.data_dir,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(**config_dict)


def load_config(config_path: str, **overrides) -> Config:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Key-value pairs to override config values
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply overrides
    config_dict.update(overrides)
    
    # Create config object
    config = Config.from_dict(config_dict)
    
    # Create directories if they don't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)


if __name__ == "__main__":
    # Test config loading
    config = load_config("configs/default.yaml")
    print("Loaded config:")
    print(config)