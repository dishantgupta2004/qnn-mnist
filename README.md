# Quantum Neural Network for MNIST

A hybrid quantum-classical neural network implementation using PennyLane and PyTorch for MNIST digit classification, with an interactive Streamlit frontend.

## Features

- **Hybrid QNN Architecture**: Quantum circuit with angle embedding + variational layers → classical linear layer
- **Fast Simulation**: Uses PennyLane's `lightning.qubit` device for efficient quantum simulation
- **Interactive UI**: Streamlit interface with real-time training visualization
- **Configurable**: YAML config + UI controls for all hyperparameters
- **Modular Design**: Clean separation of data, model, training, and evaluation logic
- **Optional PCA**: Dimensionality reduction from 784 → 8/16 features
- **Checkpointing**: Save and resume training runs

## Architecture

```
MNIST (28×28) → Flatten (784) → [Optional PCA] → AngleEmbedding (n_qubits)
    ↓
Variational Quantum Circuit (L layers):
  - Rot gates (3 params per qubit)
  - CNOT ring entanglement
    ↓
Expectation values (Z measurement) → Linear(n_qubits, 10) → Softmax
```

## Requirements

- Python 3.11
- CUDA optional (CPU simulation is default)

## Installation

```bash
# Clone or create project directory
cd mnist-qnn

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Streamlit UI (Recommended)

```bash
streamlit run app/streamlit_app.py
```

Then:
1. Adjust hyperparameters in the sidebar (qubits, layers, PCA, epochs, etc.)
2. Click **"Prepare Data"** to load and preprocess MNIST
3. Click **"Train Model"** to start training
4. View real-time loss/accuracy plots
5. Click **"Evaluate on Test Set"** to see confusion matrix
6. Click **"Predict Random Sample"** to test individual predictions

### 2. CLI Training

```bash
# Train with default config
python scripts/train_cli.py --config configs/default.yaml

# Override parameters
python scripts/train_cli.py --config configs/default.yaml --epochs 15 --lr 0.0005
```

### 3. CLI Evaluation

```bash
# Evaluate a saved checkpoint
python scripts/eval_cli.py --checkpoint data/checkpoints/qnn_model_20241031_120000.pt
```

## Project Structure

```
qnn-mnist/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml          # Default hyperparameters
├── data/
│   ├── MNIST/                # Auto-downloaded dataset
│   └── checkpoints/          # Saved model checkpoints
├── app/
│   └── streamlit_app.py      # Interactive UI
├── src/
│   ├── __init__.py
│   ├── config.py             # Config loader
│   ├── data.py               # MNIST data loading + PCA
│   ├── circuits.py           # Quantum circuit definitions
│   ├── qnn_model.py          # Hybrid QNN model
│   ├── train.py              # Training loop
│   ├── eval.py               # Evaluation utilities
│   └── utils.py              # Helpers (seeding, metrics, plots)
├── tests/
│   ├── test_config.py
│   ├── test_circuits.py
│   └── test_qnn_forward.py
└── scripts/
    ├── download_data.py      # Pre-download MNIST
    ├── train_cli.py          # Headless training
    └── eval_cli.py           # Headless evaluation
```

## Configuration

Edit `configs/default.yaml` or override via CLI/UI:

```yaml
seed: 42
n_qubits: 8           # Number of qubits
layers: 3             # Variational circuit depth
pca_k: 8              # PCA components (0 = disabled)
epochs: 10
batch_size: 64
lr: 0.001
train_size: 10000     # Subsample for fast training
val_size: 2000
device: "cpu"
shots: null           # null = analytic, or int for shot-based
checkpoint_dir: "data/checkpoints"
```

## Example Results

Typical performance with default settings (8 qubits, 3 layers, 10k training samples):

- **Training Time**: ~5-10 minutes on modern CPU
- **Validation Accuracy**: 85-92%
- **Test Accuracy**: 84-90%

Note: Results vary due to quantum circuit initialization and small training set.

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## Troubleshooting

### Issue: Slow training
- **Solution**: Reduce `train_size` (e.g., 5000) or `epochs` (e.g., 5)
- Enable PCA to reduce input dimensionality: `pca_k: 8`

### Issue: Out of memory
- **Solution**: Reduce `batch_size` (e.g., 32 or 16)

### Issue: Poor accuracy
- **Solution**: Increase `n_qubits` (e.g., 12), `layers` (e.g., 4-5), or `train_size`
- Tune learning rate: try `lr: 0.0005` or `0.002`

### Issue: Module import errors
- **Solution**: Ensure virtual environment is activated and all packages installed
- Run: `pip install -r requirements.txt --upgrade`

### Issue: MNIST download fails
- **Solution**: Pre-download using `python scripts/download_data.py`
- Or manually download to `data/MNIST/raw/`

## Advanced Usage

### Custom Quantum Circuit

Modify `src/circuits.py` to experiment with:
- Different entanglement patterns (ladder, all-to-all)
- Alternative gates (RY, RZ only)
- Observable measurements (X, Y basis)

### Hyperparameter Search

Use the Streamlit UI to quickly iterate:
1. Train with different configs
2. Compare checkpoint performance
3. Load best checkpoint for evaluation

### Export Trained Model

```python
import torch
from src.qnn_model import QuantumNet
from src.config import load_config

config = load_config("configs/default.yaml")
model = QuantumNet(config)
model.load_state_dict(torch.load("data/checkpoints/best_model.pt"))
torch.save(model.state_dict(), "exported_qnn.pt")
```

## Citation

If you use this code, please cite:

```
@software{qnn_mnist,
  title={Hybrid Quantum Neural Network for MNIST},
  author={Dishant Gupta},
  year={2025},
  url={https://github.com/dishantgupta2004/qnn-mnist}
}
```

## License

MIT License - See LICENSE file for details

## References

- [PennyLane Documentation](https://docs.pennylane.ai/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Quantum Machine Learning Papers](https://pennylane.ai/qml/)