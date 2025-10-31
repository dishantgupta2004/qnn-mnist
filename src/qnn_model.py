"""Hybrid Quantum-Classical Neural Network model."""

import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional
from src.circuits import create_quantum_circuit, get_circuit_specs


class QuantumNet(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for MNIST classification.
    
    Architecture:
        Input -> [Optional: dimension adjustment] -> Quantum Circuit -> Linear -> Output
    """
    
    def __init__(
        self,
        config,
        input_dim: Optional[int] = None
    ):
        """
        Initialize Quantum Neural Network.
        
        Args:
            config: Configuration object with n_qubits, layers, shots, etc.
            input_dim: Input feature dimension (if None, uses config.pca_k or 784)
        """
        super().__init__()
        
        self.n_qubits = config.n_qubits
        self.layers = config.layers
        self.shots = config.shots
        
        # Determine input dimension
        if input_dim is None:
            self.input_dim = config.pca_k if config.pca_k > 0 else 784
        else:
            self.input_dim = input_dim
        
        # If input_dim > n_qubits, add a linear projection layer
        if self.input_dim > self.n_qubits:
            self.input_projection = nn.Linear(self.input_dim, self.n_qubits)
        elif self.input_dim < self.n_qubits:
            raise ValueError(
                f"input_dim ({self.input_dim}) must be >= n_qubits ({self.n_qubits}). "
                f"Use PCA to reduce dimensions or decrease n_qubits."
            )
        else:
            self.input_projection = None
        
        # Create quantum circuit
        self.quantum_circuit = create_quantum_circuit(
            n_qubits=self.n_qubits,
            layers=self.layers,
            shots=self.shots
        )
        
        # Initialize quantum weights
        weight_shape = (self.layers, self.n_qubits, 3)
        self.q_weights = nn.Parameter(
            torch.randn(weight_shape) * 0.01
        )
        
        # Classical output layer
        self.fc_out = nn.Linear(self.n_qubits, 10)
        
        # Store circuit specs
        self.circuit_specs = get_circuit_specs(self.n_qubits, self.layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid QNN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Flatten if image format
        if x.dim() == 4:  # (B, 1, 28, 28)
            x = x.view(x.size(0), -1)  # (B, 784)
        
        # Project to quantum dimension if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
            x = torch.tanh(x)  # Normalize to [-1, 1] for AngleEmbedding
        
        # Process through quantum circuit (batch processing)
        batch_size = x.size(0)
        quantum_out = []
        
        for i in range(batch_size):
            q_out = self.quantum_circuit(x[i], self.q_weights)
            # Convert list to tensor if needed
            if isinstance(q_out, list):
                q_out = torch.stack(q_out)
            quantum_out.append(q_out)
        
        quantum_out = torch.stack(quantum_out)  # (batch_size, n_qubits)
        
        # Classical output layer
        output = self.fc_out(quantum_out)  # (batch_size, 10)
        
        return output
    
    def get_num_parameters(self) -> dict:
        """Get number of trainable parameters in each component."""
        quantum_params = self.q_weights.numel()
        classical_params = sum(p.numel() for p in self.fc_out.parameters())
        
        if self.input_projection is not None:
            projection_params = sum(p.numel() for p in self.input_projection.parameters())
        else:
            projection_params = 0
        
        total_params = quantum_params + classical_params + projection_params
        
        return {
            "quantum_params": quantum_params,
            "classical_params": classical_params,
            "projection_params": projection_params,
            "total_params": total_params
        }
    
    def describe(self):
        """Print model architecture description."""
        print("\n" + "="*60)
        print("HYBRID QUANTUM NEURAL NETWORK")
        print("="*60)
        print(f"\nInput dimension: {self.input_dim}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Quantum layers: {self.layers}")
        print(f"Shot mode: {'Analytic' if self.shots is None else f'{self.shots} shots'}")
        
        if self.input_projection is not None:
            print(f"\nInput projection: {self.input_dim} -> {self.n_qubits}")
        
        print("\nQuantum Circuit:")
        for key, value in self.circuit_specs.items():
            print(f"  {key}: {value}")
        
        print("\nClassical output layer: Linear({} -> 10)".format(self.n_qubits))
        
        params = self.get_num_parameters()
        print("\nTrainable Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value:,}")
        
        print("="*60 + "\n")


def create_model(config, input_dim: Optional[int] = None) -> QuantumNet:
    """
    Factory function to create a QuantumNet model.
    
    Args:
        config: Configuration object
        input_dim: Optional input dimension override
        
    Returns:
        Initialized QuantumNet model
    """
    model = QuantumNet(config, input_dim)
    return model


