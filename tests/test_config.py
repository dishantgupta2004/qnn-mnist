"""Tests for configuration module."""

import pytest
import tempfile
import os
from pathlib import Path
from src.config import Config, load_config, save_config


def test_config_creation():
    """Test basic config creation."""
    config = Config()
    assert config.seed == 42
    assert config.n_qubits == 8
    assert config.layers == 3


def test_config_validation_qubits():
    """Test config validation for n_qubits."""
    with pytest.raises(ValueError):
        Config(n_qubits=0)


def test_config_validation_layers():
    """Test config validation for layers."""
    with pytest.raises(ValueError):
        Config(layers=0)


def test_config_validation_pca():
    """Test config validation for PCA."""
    with pytest.raises(ValueError):
        Config(n_qubits=8, pca_k=4)  # pca_k < n_qubits


def test_config_to_dict():
    """Test config conversion to dictionary."""
    config = Config(n_qubits=4, layers=2)
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict['n_qubits'] == 4
    assert config_dict['layers'] == 2


def test_config_from_dict():
    """Test config creation from dictionary."""
    config_dict = {
        'seed': 123,
        'n_qubits': 6,
        'layers': 2,
        'pca_k': 8,
        'epochs': 5,
        'batch_size': 32,
        'lr': 0.0005,
        'train_size': 5000,
        'val_size': 1000,
        'device': 'cpu',
        'shots': None,
        'checkpoint_dir': 'data/checkpoints',
        'data_dir': 'data'
    }
    
    config = Config.from_dict(config_dict)
    assert config.seed == 123
    assert config.n_qubits == 6
    assert config.lr == 0.0005


def test_load_config():
    """Test loading config from YAML file."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
seed: 99
n_qubits: 4
layers: 2
pca_k: 8
epochs: 3
batch_size: 16
lr: 0.0001
train_size: 1000
val_size: 200
device: "cpu"
shots: null
checkpoint_dir: "data/checkpoints"
data_dir: "data"
""")
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config.seed == 99
        assert config.n_qubits == 4
        assert config.batch_size == 16
    finally:
        os.unlink(temp_path)


def test_load_config_with_overrides():
    """Test loading config with overrides."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
seed: 42
n_qubits: 8
layers: 3
pca_k: 8
epochs: 10
batch_size: 64
lr: 0.001
train_size: 10000
val_size: 2000
device: "cpu"
shots: null
checkpoint_dir: "data/checkpoints"
data_dir: "data"
""")
        temp_path = f.name
    
    try:
        config = load_config(temp_path, epochs=5, lr=0.0005)
        assert config.epochs == 5
        assert config.lr == 0.0005
        assert config.n_qubits == 8  # Original value
    finally:
        os.unlink(temp_path)


def test_save_config():
    """Test saving config to YAML file."""
    config = Config(n_qubits=4, layers=2, epochs=5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_config.yaml')
        save_config(config, save_path)
        
        # Load it back
        loaded_config = load_config(save_path)
        assert loaded_config.n_qubits == 4
        assert loaded_config.layers == 2
        assert loaded_config.epochs == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])