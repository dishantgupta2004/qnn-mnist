"""Tests for quantum circuits."""

import pytest
import torch
import numpy as np
from src.circuits import create_quantum_circuit, get_circuit_specs, draw_circuit


def test_circuit_creation():
    """Test basic circuit creation."""
    n_qubits = 4
    layers = 2
    
    circuit = create_quantum_circuit(n_qubits, layers)
    assert circuit is not None


def test_circuit_forward_pass():
    """Test circuit forward pass with valid inputs."""
    n_qubits = 4
    layers = 2
    
    circuit = create_quantum_circuit(n_qubits, layers)
    
    # Create inputs
    features = torch.randn(n_qubits)
    weights = torch.randn(layers, n_qubits, 3)
    
    # Forward pass
    output = circuit(features, weights)
    
    # Check output
    assert output is not None
    assert len(output) == n_qubits
    
    # Convert to tensor if list
    if isinstance(output, list):
        output = torch.stack(output)
    
    # Check all values are in [-1, 1] (expectation values)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_circuit_different_sizes():
    """Test circuits with different sizes."""
    test_cases = [
        (2, 1),
        (4, 2),
        (8, 3),
        (16, 4)
    ]
    
    for n_qubits, layers in test_cases:
        circuit = create_quantum_circuit(n_qubits, layers)
        
        features = torch.randn(n_qubits)
        weights = torch.randn(layers, n_qubits, 3)
        
        output = circuit(features, weights)
        
        if isinstance(output, list):
            assert len(output) == n_qubits
        else:
            assert output.shape[0] == n_qubits


def test_circuit_gradient_flow():
    """Test that gradients can flow through the circuit."""
    n_qubits = 4
    layers = 2
    
    circuit = create_quantum_circuit(n_qubits, layers)
    
    features = torch.randn(n_qubits)
    weights = torch.randn(layers, n_qubits, 3, requires_grad=True)
    
    output = circuit(features, weights)
    
    if isinstance(output, list):
        output = torch.stack(output)
    
    # Compute loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert weights.grad is not None
    assert not torch.all(weights.grad == 0)


def test_circuit_with_shots():
    """Test circuit with shot-based simulation."""
    n_qubits = 4
    layers = 2
    shots = 1000
    
    circuit = create_quantum_circuit(n_qubits, layers, shots=shots)
    
    features = torch.randn(n_qubits)
    weights = torch.randn(layers, n_qubits, 3)
    
    output = circuit(features, weights)
    
    assert output is not None
    if isinstance(output, list):
        assert len(output) == n_qubits
    else:
        assert output.shape[0] == n_qubits


def test_get_circuit_specs():
    """Test circuit specifications calculation."""
    specs = get_circuit_specs(n_qubits=8, layers=3)
    
    assert specs['n_qubits'] == 8
    assert specs['layers'] == 3
    assert specs['n_rot_gates'] == 24  # 8 qubits * 3 layers
    assert specs['n_cnot_gates'] == 24  # 8 qubits * 3 layers
    assert specs['n_trainable_params'] == 72  # 8 * 3 * 3
    assert specs['circuit_depth'] == 6  # 2 * 3 layers


def test_draw_circuit_text():
    """Test circuit drawing in text format."""
    n_qubits = 4
    layers = 2
    
    circuit = create_quantum_circuit(n_qubits, layers)
    drawing = draw_circuit(circuit, n_qubits, layers, format="text")
    
    assert isinstance(drawing, str)
    assert len(drawing) > 0


def test_circuit_deterministic():
    """Test that circuit is deterministic with same inputs."""
    n_qubits = 4
    layers = 2
    
    circuit = create_quantum_circuit(n_qubits, layers)
    
    torch.manual_seed(42)
    features = torch.randn(n_qubits)
    weights = torch.randn(layers, n_qubits, 3)
    
    output1 = circuit(features, weights)
    output2 = circuit(features, weights)
    
    if isinstance(output1, list):
        output1 = torch.stack(output1)
        output2 = torch.stack(output2)
    
    assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])