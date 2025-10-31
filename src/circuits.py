"""Quantum circuit definitions using PennyLane."""

import pennylane as qml
import numpy as np
from typing import Optional


def create_quantum_circuit(
    n_qubits: int,
    layers: int,
    shots: Optional[int] = None
) -> qml.QNode:
    """
    Create a variational quantum circuit with angle embedding and entanglement.
    
    Architecture:
        1. AngleEmbedding: Encode classical data into rotation angles
        2. Variational layers: 
           - Rot gates (3 parameters per qubit)
           - CNOT ring entanglement
        3. Measurement: Expectation values of Pauli-Z on each qubit
    
    Args:
        n_qubits: Number of qubits in the circuit
        layers: Number of variational layers
        shots: Number of shots for measurement (None for analytic)
        
    Returns:
        QNode that takes (features, weights) and returns expectations
    """
    # Create device
    dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)
    
    @qml.qnode(dev, interface="torch")
    def circuit(features, weights):
        """
        Quantum circuit.
        
        Args:
            features: Input features (n_qubits,)
            weights: Trainable parameters (layers, n_qubits, 3)
            
        Returns:
            Expectation values (n_qubits,)
        """
        # Encode features using angle embedding
        qml.AngleEmbedding(features, wires=range(n_qubits))
        
        # Variational layers
        for layer_idx in range(layers):
            # Apply parametrized rotation gates
            for wire in range(n_qubits):
                qml.Rot(
                    weights[layer_idx, wire, 0],
                    weights[layer_idx, wire, 1],
                    weights[layer_idx, wire, 2],
                    wires=wire
                )
            
            # Apply entangling CNOT gates in a ring topology
            for wire in range(n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
        
        # Measure expectation values
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
    
    return circuit


def draw_circuit(
    circuit: qml.QNode,
    n_qubits: int,
    layers: int,
    format: str = "text"
) -> str:
    """
    Draw the quantum circuit.
    
    Args:
        circuit: QNode to draw
        n_qubits: Number of qubits
        layers: Number of layers
        format: Output format ("text" or "mpl")
        
    Returns:
        Circuit drawing as string (text format) or None (matplotlib format)
    """
    # Create dummy inputs
    dummy_features = np.zeros(n_qubits)
    dummy_weights = np.zeros((layers, n_qubits, 3))
    
    if format == "text":
        return qml.draw(circuit)(dummy_features, dummy_weights)
    elif format == "mpl":
        fig, ax = qml.draw_mpl(circuit)(dummy_features, dummy_weights)
        return fig
    else:
        raise ValueError(f"Unknown format: {format}. Use 'text' or 'mpl'")


def get_circuit_specs(n_qubits: int, layers: int) -> dict:
    """
    Get circuit specifications.
    
    Args:
        n_qubits: Number of qubits
        layers: Number of layers
        
    Returns:
        Dictionary with circuit specs
    """
    # Count gates
    n_rot_gates = layers * n_qubits
    n_cnot_gates = layers * n_qubits
    n_params = layers * n_qubits * 3  # 3 params per Rot gate
    
    return {
        "n_qubits": n_qubits,
        "layers": layers,
        "n_rot_gates": n_rot_gates,
        "n_cnot_gates": n_cnot_gates,
        "total_gates": n_rot_gates + n_cnot_gates,
        "n_trainable_params": n_params,
        "circuit_depth": 2 * layers,  # Rot + CNOT per layer
    }


