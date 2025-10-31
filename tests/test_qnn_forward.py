"""Tests for QNN model forward pass."""

import pytest
import torch
from src.config import Config
from src.qnn_model import QuantumNet, create_model


def test_qnn_creation():
    """Test QNN model creation."""
    config = Config(n_qubits=8, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    assert model is not None
    assert model.n_qubits == 8
    assert model.layers == 2


def test_qnn_forward_with_pca():
    """Test QNN forward pass with PCA input."""
    config = Config(n_qubits=8, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    batch_size = 4
    x = torch.randn(batch_size, 8)
    
    output = model(x)
    
    assert output.shape == (batch_size, 10)


def test_qnn_forward_with_images():
    """Test QNN forward pass with image input."""
    config = Config(n_qubits=8, layers=2, pca_k=0)
    model = create_model(config, input_dim=784)
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    
    output = model(x)
    
    assert output.shape == (batch_size, 10)


def test_qnn_forward_flattened():
    """Test QNN forward pass with flattened input."""
    config = Config(n_qubits=8, layers=2, pca_k=0)
    model = create_model(config, input_dim=784)
    
    batch_size = 4
    x = torch.randn(batch_size, 784)
    
    output = model(x)
    
    assert output.shape == (batch_size, 10)


def test_qnn_backward():
    """Test QNN backward pass (gradient computation)."""
    config = Config(n_qubits=4, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    x = torch.randn(2, 8)
    target = torch.tensor([0, 5])
    
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    
    loss.backward()
    
    # Check that gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_qnn_parameter_count():
    """Test parameter counting."""
    config = Config(n_qubits=8, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    params = model.get_num_parameters()
    
    assert params['quantum_params'] == 8 * 2 * 3  # n_qubits * layers * 3
    assert params['classical_params'] == 8 * 10 + 10  # weights + bias
    assert params['total_params'] > 0


def test_qnn_with_projection():
    """Test QNN with input projection layer."""
    config = Config(n_qubits=8, layers=2, pca_k=16)
    model = create_model(config, input_dim=16)
    
    assert model.input_projection is not None
    
    x = torch.randn(4, 16)
    output = model(x)
    
    assert output.shape == (4, 10)


def test_qnn_different_batch_sizes():
    """Test QNN with different batch sizes."""
    config = Config(n_qubits=8, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    for batch_size in [1, 2, 8, 16]:
        x = torch.randn(batch_size, 8)
        output = model(x)
        assert output.shape == (batch_size, 10)


def test_qnn_eval_mode():
    """Test QNN in evaluation mode."""
    config = Config(n_qubits=8, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    model.eval()
    
    x = torch.randn(4, 8)
    output = model(x)
    
    assert output.shape == (4, 10)


def test_qnn_train_mode():
    """Test QNN in training mode."""
    config = Config(n_qubits=8, layers=2, pca_k=8)
    model = create_model(config, input_dim=8)
    
    model.train()
    
    x = torch.randn(4, 8)
    output = model(x)
    
    assert output.shape == (4, 10)


def test_qnn_invalid_input_dim():
    """Test QNN with invalid input dimension."""
    config = Config(n_qubits=8, layers=2, pca_k=4)
    
    # Should raise error because pca_k < n_qubits
    with pytest.raises(ValueError):
        model = create_model(config, input_dim=4)


def test_qnn_circuit_specs():
    """Test QNN circuit specifications."""
    config = Config(n_qubits=8, layers=3, pca_k=8)
    model = create_model(config, input_dim=8)
    
    specs = model.circuit_specs
    
    assert specs['n_qubits'] == 8
    assert specs['layers'] == 3
    assert specs['n_trainable_params'] == 8 * 3 * 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])