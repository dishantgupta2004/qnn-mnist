"""Streamlit app for Quantum Neural Network training and evaluation."""

import streamlit as st
import torch
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data import get_data_loaders
from src.qnn_model import create_model
from src.train import train_model
from src.eval import evaluate_model, predict_single
from src.utils import (
    seed_everything, plot_training_curves, plot_confusion_matrix,
    plot_sample_predictions, format_time
)
from src.circuits import draw_circuit


# Page config
st.set_page_config(
    page_title="Quantum Neural Network - MNIST",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Hybrid Quantum Neural Network for MNIST")
st.markdown("Train and evaluate a quantum-classical neural network using PennyLane and PyTorch")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'history' not in st.session_state:
    st.session_state.history = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'input_dim' not in st.session_state:
    st.session_state.input_dim = None

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Quantum circuit parameters
st.sidebar.subheader("Quantum Circuit")
n_qubits = st.sidebar.slider("Number of Qubits", 4, 16, 8)
layers = st.sidebar.slider("Variational Layers", 1, 5, 3)

# Data preprocessing
st.sidebar.subheader("Data Preprocessing")
use_pca = st.sidebar.checkbox("Use PCA", value=True)
pca_k = st.sidebar.slider("PCA Components", 4, 16, 8) if use_pca else 0

# Training parameters
st.sidebar.subheader("Training")
epochs = st.sidebar.slider("Epochs", 1, 20, 10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
lr = st.sidebar.select_slider("Learning Rate", 
                                options=[0.0001, 0.0005, 0.001, 0.002, 0.005],
                                value=0.001)

# Dataset size
st.sidebar.subheader("Dataset")
train_size = st.sidebar.slider("Training Samples", 1000, 60000, 10000, step=1000)
val_size = st.sidebar.slider("Validation Samples", 500, 10000, 2000, step=500)

# Device and shots
st.sidebar.subheader("Device")
device_option = st.sidebar.selectbox("Device", ["CPU", "CUDA"])
device = "cuda" if device_option == "CUDA" and torch.cuda.is_available() else "cpu"

shot_mode = st.sidebar.radio("Measurement Mode", ["Analytic", "Shot-based"])
shots = None if shot_mode == "Analytic" else st.sidebar.number_input("Number of Shots", 100, 10000, 1000)

# Other settings
st.sidebar.subheader("Other")
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

# Create config
config = Config(
    seed=seed,
    n_qubits=n_qubits,
    layers=layers,
    pca_k=pca_k,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    train_size=train_size,
    val_size=val_size,
    device=device,
    shots=shots
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Training", "🧪 Evaluation", "🎯 Prediction", "ℹ️ Info"])

with tab1:
    st.header("Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prepare data button
        if st.button("📥 Prepare Data", type="primary"):
            with st.spinner("Loading MNIST dataset..."):
                try:
                    seed_everything(config.seed)
                    
                    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
                        data_dir=config.data_dir,
                        train_size=config.train_size,
                        val_size=config.val_size,
                        batch_size=config.batch_size,
                        pca_k=config.pca_k,
                        seed=config.seed
                    )
                    
                    st.session_state.train_loader = train_loader
                    st.session_state.val_loader = val_loader
                    st.session_state.test_loader = test_loader
                    st.session_state.input_dim = input_dim
                    st.session_state.data_loaded = True
                    st.session_state.config = config
                    
                    st.success(f"✓ Data loaded! Input dimension: {input_dim}")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Train button
        if st.button("🚀 Train Model", type="primary", disabled=not st.session_state.data_loaded):
            with st.spinner("Training model... This may take several minutes."):
                try:
                    # Create model
                    model = create_model(config, input_dim=st.session_state.input_dim)
                    
                    # Display model info
                    st.info(f"Model created with {model.get_num_parameters()['total_params']:,} parameters")
                    
                    # Create checkpoint path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = os.path.join(
                        config.checkpoint_dir,
                        f"qnn_model_{timestamp}.pt"
                    )
                    os.makedirs(config.checkpoint_dir, exist_ok=True)
                    
                    # Train
                    start_time = time.time()
                    history = train_model(
                        model=model,
                        train_loader=st.session_state.train_loader,
                        val_loader=st.session_state.val_loader,
                        config=config,
                        device=device,
                        checkpoint_path=checkpoint_path
                    )
                    train_time = time.time() - start_time
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.history = history
                    st.session_state.model_trained = True
                    st.session_state.checkpoint_path = checkpoint_path
                    
                    st.success(f"✓ Training complete in {format_time(train_time)}!")
                    st.success(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
                    st.info(f"Checkpoint saved: {checkpoint_path}")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        st.subheader("Configuration Summary")
        st.json({
            "n_qubits": config.n_qubits,
            "layers": config.layers,
            "pca_k": config.pca_k,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "train_size": config.train_size,
            "val_size": config.val_size,
            "device": config.device
        })
    
    # Plot training curves
    if st.session_state.history:
        st.subheader("Training History")
        fig = plot_training_curves(st.session_state.history, show=False)
        st.pyplot(fig)
        
        # Display metrics table
        st.subheader("Epoch Metrics")
        import pandas as pd
        df = pd.DataFrame({
            'Epoch': range(1, len(st.session_state.history['train_loss']) + 1),
            'Train Loss': st.session_state.history['train_loss'],
            'Val Loss': st.session_state.history['val_loss'],
            'Train Acc': st.session_state.history['train_acc'],
            'Val Acc': st.session_state.history['val_acc'],
        })
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("Evaluation")
    
    if st.button("🧪 Evaluate on Test Set", disabled=not st.session_state.model_trained):
        with st.spinner("Evaluating model on test set..."):
            try:
                results = evaluate_model(
                    st.session_state.model,
                    st.session_state.test_loader,
                    device=device
                )
                
                st.session_state.eval_results = results
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{results['accuracy']:.2f}%")
                with col2:
                    st.metric("Total Samples", len(results['labels']))
                with col3:
                    correct = (results['predictions'] == results['labels']).sum()
                    st.metric("Correct Predictions", correct)
                
                st.success("✓ Evaluation complete!")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
    
    # Show confusion matrix
    if 'eval_results' in st.session_state:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(
            st.session_state.eval_results['labels'],
            st.session_state.eval_results['predictions'],
            show=False
        )
        st.pyplot(fig)
        
        # Per-class metrics
        st.subheader("Per-Class Performance")
        import pandas as pd
        report = st.session_state.eval_results['classification_report']
        
        data = []
        for i in range(10):
            class_name = str(i)
            if class_name in report:
                metrics = report[class_name]
                data.append({
                    'Class': i,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1-score']:.4f}",
                    'Support': int(metrics['support'])
                })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

with tab3:
    st.header("Prediction")
    
    if st.button("🎯 Predict Random Sample", disabled=not st.session_state.model_trained):
        with st.spinner("Making prediction..."):
            try:
                # Get random sample from test set
                data_iter = iter(st.session_state.test_loader)
                images, labels = next(data_iter)
                
                # Pick random index
                idx = np.random.randint(0, len(images))
                image = images[idx]
                true_label = labels[idx].item()
                
                # Predict
                pred_class, probabilities = predict_single(
                    st.session_state.model,
                    image,
                    device=device
                )
                
                # Display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Show image
                    if image.dim() == 3 and image.shape[0] == 1:
                        img_display = image.squeeze().numpy()
                    elif image.dim() == 1:
                        # PCA case - get from original test set
                        st.warning("PCA features - original image not available")
                        img_display = None
                    else:
                        img_display = image.numpy()
                    
                    if img_display is not None:
                        st.image(img_display, caption=f"True Label: {true_label}", 
                                width=200, clamp=True)
                
                with col2:
                    # Show prediction
                    if pred_class == true_label:
                        st.success(f"✓ Correct Prediction: {pred_class}")
                    else:
                        st.error(f"✗ Incorrect Prediction: {pred_class} (True: {true_label})")
                    
                    # Show probabilities
                    st.subheader("Class Probabilities")
                    prob_dict = {f"Class {i}": f"{probabilities[i].item():.4f}" 
                                for i in range(10)}
                    st.json(prob_dict)
                    
                    # Bar chart
                    import pandas as pd
                    df = pd.DataFrame({
                        'Class': range(10),
                        'Probability': probabilities.detach().cpu().numpy()
                    })
                    st.bar_chart(df.set_index('Class'))
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # Sample predictions visualization
    if st.button("📊 Show Sample Predictions (10 images)", 
                 disabled=not st.session_state.model_trained):
        with st.spinner("Generating predictions..."):
            try:
                # Get batch
                data_iter = iter(st.session_state.test_loader)
                images, labels = next(data_iter)
                
                # Predict
                st.session_state.model.eval()
                with torch.no_grad():
                    outputs = st.session_state.model(images[:10].to(device))
                    _, predictions = torch.max(outputs, 1)
                
                # Plot
                fig = plot_sample_predictions(
                    images[:10],
                    labels[:10],
                    predictions.cpu(),
                    n_samples=10,
                    show=False
                )
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

with tab4:
    st.header("Information")
    
    # Model architecture
    if st.session_state.model:
        st.subheader("Model Architecture")
        params = st.session_state.model.get_num_parameters()
        specs = st.session_state.model.circuit_specs
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Parameters:**")
            st.json(params)
        
        with col2:
            st.markdown("**Circuit Specifications:**")
            st.json(specs)
        
        # Circuit diagram
        st.subheader("Quantum Circuit")
        try:
            from src.circuits import create_quantum_circuit
            circuit = create_quantum_circuit(config.n_qubits, config.layers, config.shots)
            circuit_str = draw_circuit(circuit, config.n_qubits, config.layers, format="text")
            st.code(circuit_str)
        except Exception as e:
            st.error(f"Error drawing circuit: {str(e)}")
    
    # About
    st.subheader("About")
    st.markdown("""
    This application demonstrates a **Hybrid Quantum-Classical Neural Network** for MNIST digit classification.
    
    **Architecture:**
    - Input: MNIST images (28×28 pixels)
    - Optional PCA for dimensionality reduction
    - Quantum circuit with angle embedding and variational layers
    - Classical linear layer for final classification
    
    **Technologies:**
    - PennyLane: Quantum machine learning framework
    - PyTorch: Neural network training
    - Streamlit: Interactive web interface
    
    **Tips:**
    - Start with default parameters for quick experiments
    - Use PCA to speed up training (reduces input dimension)
    - Increase epochs and training samples for better accuracy
    - Try different quantum circuit architectures (qubits, layers)
    """)
    
    # Links
    st.subheader("Resources")
    st.markdown("""
    - [PennyLane Documentation](https://docs.pennylane.ai/)
    - [PyTorch Tutorials](https://pytorch.org/tutorials/)
    - [Quantum Machine Learning](https://pennylane.ai/qml/)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🔬 **Quantum Neural Network v0.1.0**")
st.sidebar.markdown("Built with PennyLane & PyTorch")