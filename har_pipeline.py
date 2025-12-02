#!/usr/bin/env python3
"""
Accelerometer-Based Motion Detection

Authors: Filip Izworski, Jakub Kozdrój, Mikołaj Kołodziej
Date: 2.12.2025

Complete pipeline for Human Activity Recognition model compression.
Features:
- Training lightweight 1D-CNN on UCI HAR dataset
- Model compression via pruning (sparsity) and quantization (INT8/INT4)
- Sparse format storage for actual size reduction
- Comparison of compression techniques vs FP32 baseline
- Visualization of accuracy/size trade-offs

Usage examples:
  # Baseline only
  python har_pipeline.py
  
  # Pruning only (30%)
  python har_pipeline.py --prune
  
  # Quantization only (INT8)
  python har_pipeline.py --quantize
  
  # Both techniques (30% pruning + INT8)
  python har_pipeline.py --prune --quantize
  
  # Custom parameters (50% pruning + INT4)
  python har_pipeline.py --prune --prune_amount 0.5 --quantize --quantize_bits 4
"""

import os
import zipfile
import argparse
from urllib.request import urlretrieve
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATA_DIR = "UCI_HAR_Dataset"
RESULTS_DIR = "results"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def download_dataset():
    """Download UCI HAR dataset if not present locally."""
    if os.path.exists(DATA_DIR):
        print("Dataset already exists.")
        return
    
    print("Downloading dataset...")
    urlretrieve(DATA_URL, "temp.zip")
    
    with zipfile.ZipFile("temp.zip", "r") as z:
        z.extractall(".")
    
    # Rename to consistent folder name
    if os.path.exists("UCI HAR Dataset"):
        os.rename("UCI HAR Dataset", DATA_DIR)
    
    os.remove("temp.zip")
    print("Dataset ready.")


def load_signals(subset):
    """Load 6-channel sensor signals from text files."""
    signal_names = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
                   "body_gyro_x_", "body_gyro_y_", "body_gyro_z_"]
    
    signals = []
    for name in signal_names:
        path = os.path.join(DATA_DIR, subset, "Inertial Signals", name + subset + ".txt")
        signals.append(np.loadtxt(path))
    
    # Transpose to (samples, channels, timesteps)
    return np.transpose(np.array(signals), (1, 0, 2))


def load_labels(subset):
    """Load activity labels (0-5 for 6 activities)."""
    path = os.path.join(DATA_DIR, subset, "y_" + subset + ".txt")
    return (np.loadtxt(path).astype(int) - 1).astype(int)


def prepare_data():
    """Prepare PyTorch DataLoaders with per-channel normalization."""
    # Load raw data
    X_train = load_signals("train")
    X_test = load_signals("test")
    y_train = load_labels("train")
    y_test = load_labels("test")
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Normalize each channel independently
    n_samples, n_channels, timesteps = X_train.shape
    scaler = StandardScaler()
    
    X_train_norm = scaler.fit_transform(X_train.reshape(-1, n_channels)).reshape(X_train.shape)
    X_test_norm = scaler.transform(X_test.reshape(-1, n_channels)).reshape(X_test.shape)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, timesteps


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class HAR_CNN(nn.Module):
    """
    Lightweight 1D CNN for Human Activity Recognition.
    
    Architecture:
    - Input: (batch, 6, 128) - 6 sensor channels, 128 timesteps
    - Conv1D: 32 filters, kernel=5, padding=2
    - ReLU + MaxPool1D(2)
    - Conv1D: 64 filters, kernel=5, padding=2
    - ReLU + Global Average Pooling
    - Fully Connected: 64 -> 6 classes
    
    Total parameters: ~13,000
    """
    def __init__(self, in_channels=6, n_classes=6):
        super(HAR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, n_classes)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.fc(x)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader):
    """Train model for specified number of epochs."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


def evaluate(model, dataloader):
    """Calculate accuracy on test dataset."""
    model.to(DEVICE)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total


# ============================================================================
# PRUNING FUNCTIONS
# ============================================================================

def apply_pruning(model, amount=0.3, conv_only=True):
    """
    Apply magnitude-based pruning to model weights.
    
    Args:
        model: PyTorch model to prune
        amount: Fraction of weights to zero out (0.0 to 1.0)
        conv_only: If True, only prune convolutional layers
    
    Returns:
        tuple: (pruned_model, actual_sparsity, non_zero_count)
    """
    print(f"Applying pruning (amount: {amount:.0%})...")
    print(f"  Note: Light pruning can improve accuracy via regularization.")
    
    # Identify layers to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            parameters_to_prune.append((module, 'weight'))
        elif not conv_only and isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply L1 unstructured pruning
    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=amount)
    
    # Make pruning permanent (remove mask, keep zeros)
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    # Calculate sparsity statistics
    total_params = 0
    zero_params = 0
    non_zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_count = (param == 0).sum().item()
            zero_params += zero_count
            non_zero_params += param.numel() - zero_count
    
    actual_sparsity = zero_params / total_params if total_params > 0 else 0
    
    print(f"\n  Sparsity statistics:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Zero parameters: {zero_params:,} ({actual_sparsity:.1%})")
    print(f"    Non-zero parameters: {non_zero_params:,} ({(1-actual_sparsity):.1%})")
    print(f"    Size vs dense: {(non_zero_params/total_params)*100:.1f}%")
    
    return model, actual_sparsity, non_zero_params


def save_pruned_model_sparse(model, filepath, sparsity):
    """
    Save pruned model in sparse format (COO-like).
    
    Stores only non-zero values and their indices, plus original shape.
    This achieves actual file size reduction vs storing zeros.
    
    Returns:
        tuple: (filepath, size_percentage)
    """
    print(f"Saving pruned model in sparse format...")
    
    sparse_data = {}
    total_original = 0
    total_sparse = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Extract non-zero elements and their positions
            non_zero_mask = param != 0
            non_zero_values = param[non_zero_mask]
            non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False)
            
            original_size = param.numel()
            sparse_size = non_zero_values.numel()
            
            total_original += original_size
            total_sparse += sparse_size
            
            # Store sparse representation
            sparse_data[f'{name}_values'] = non_zero_values.detach().cpu().numpy().astype(np.float32)
            sparse_data[f'{name}_indices'] = non_zero_indices.detach().cpu().numpy().astype(np.uint16)
            sparse_data[f'{name}_shape'] = np.array(param.shape, dtype=np.uint16)
            
            print(f"  {name}: {original_size} → {sparse_size} values ({(sparse_size/original_size)*100:.1f}%)")
        else:
            # Store biases normally (dense)
            sparse_data[name] = param.detach().cpu().numpy().astype(np.float32)
    
    # Add metadata
    sparse_data['sparsity'] = np.float32(sparsity)
    sparse_data['size_vs_dense'] = np.float32(total_sparse / total_original if total_original > 0 else 1.0)
    
    # Save as compressed numpy file
    np.savez_compressed(filepath, **sparse_data)
    
    size_percentage = (total_sparse / total_original * 100) if total_original > 0 else 100
    print(f"  Overall: {total_original:,} → {total_sparse:,} values ({size_percentage:.1f}%)")
    print(f"  Compression ratio: {total_original/total_sparse:.1f}x")
    
    return filepath, size_percentage


def load_pruned_model_sparse(filepath):
    """Reconstruct model from sparse format file."""
    data = np.load(filepath, allow_pickle=True)
    
    # Create empty model
    model = HAR_CNN()
    state_dict = model.state_dict()
    
    # Reconstruct sparse tensors
    for name in list(state_dict.keys()):
        if f'{name}_values' in data:
            # Get sparse components
            values = torch.from_numpy(data[f'{name}_values']).float()
            indices = torch.from_numpy(data[f'{name}_indices']).long()
            shape = data[f'{name}_shape']
            
            # Recreate dense tensor from sparse representation
            dense_tensor = torch.zeros(tuple(shape))
            dense_tensor[tuple(indices.T)] = values
            
            state_dict[name] = dense_tensor
        elif name in data:
            # Load bias normally
            state_dict[name] = torch.from_numpy(data[name]).float()
    
    model.load_state_dict(state_dict)
    return model


# ============================================================================
# QUANTIZATION FUNCTIONS
# ============================================================================

def apply_quantization(model, precision_bits=8):
    """
    Apply symmetric quantization to model weights.
    
    Args:
        model: PyTorch model to quantize
        precision_bits: 4, 8, or 16 bits per weight
    
    Returns:
        tuple: (quantized_model, weights_dict, scales_dict, size_percentage)
    """
    if precision_bits not in [4, 8, 16]:
        raise ValueError("precision_bits must be 4, 8, or 16")
    
    print(f"Applying {precision_bits}-bit quantization...")
    
    quantized_weights = {}
    scales = {}
    
    # Quantization parameters based on bit width
    if precision_bits == 4:
        qmin, qmax = -7, 7  # 4-bit symmetric
        bytes_per_param = 0.5
    elif precision_bits == 8:
        qmin, qmax = -127, 127  # 8-bit symmetric
        bytes_per_param = 1.0
    else:  # 16-bit
        qmin, qmax = -32767, 32767  # 16-bit symmetric
        bytes_per_param = 2.0
    
    # Create copy to avoid modifying original
    quantized_model = HAR_CNN()
    quantized_model.load_state_dict(model.state_dict())
    
    # Count parameters for size calculation
    total_params = 0
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                
            # Skip biases (keep FP32)
            if 'bias' in name:
                continue
                
            # Symmetric quantization: find scale factor
            abs_max = param.abs().max().item()
            scale = abs_max / qmax if abs_max > 0 else 1.0
            
            # Quantize and clamp to integer range
            quantized = torch.round(param / scale).clamp(qmin, qmax)
            
            # Store for saving
            quantized_weights[name] = quantized.detach().cpu().numpy().astype(
                np.int8 if precision_bits == 8 else np.int16
            )
            scales[name] = np.float32(scale)
            
            # Replace with dequantized version for evaluation
            param.data.copy_(quantized.float() * scale)
    
    # Calculate theoretical size vs FP32
    total_bytes = total_params * bytes_per_param
    fp32_bytes = total_params * 4  # FP32 uses 4 bytes per parameter
    size_percentage = (total_bytes / fp32_bytes * 100)
    
    print(f"  Theoretical size vs FP32: {size_percentage:.1f}%")
    print(f"  ({total_bytes/1024:.1f} KB vs {fp32_bytes/1024:.1f} KB)")
    
    return quantized_model, quantized_weights, scales, size_percentage


def save_quantized_model(quantized_weights, scales, precision_bits, filepath):
    """Save quantized weights and scales to compressed numpy file."""
    save_dict = {
        'int_weights': quantized_weights,
        'scales': scales,
        'precision_bits': precision_bits,
        'model_info': {'in_channels': 6, 'n_classes': 6}
    }
    
    np.savez_compressed(filepath, **save_dict)


def load_quantized_model(filepath):
    """Load and reconstruct quantized model."""
    data = np.load(filepath, allow_pickle=True)
    int_weights = data['int_weights'].item()
    scales = data['scales'].item()
    precision_bits = int(data['precision_bits'])
    
    # Create model and dequantize weights
    model = HAR_CNN()
    state_dict = model.state_dict()
    
    for name in state_dict.keys():
        if name in int_weights:
            int_data = torch.from_numpy(int_weights[name]).float()
            scale = scales[name]
            state_dict[name] = int_data * scale  # Dequantize
    
    model.load_state_dict(state_dict)
    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_model_dense(model, filepath, model_name):
    """Save model in standard PyTorch format."""
    print(f"Saving {model_name}...")
    torch.save(model.state_dict(), filepath)
    return filepath


def get_model_size(filepath):
    """Get file size in bytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """
    Main execution pipeline.
    
    Steps:
    1. Prepare dataset
    2. Train FP32 baseline
    3. Apply selected compression
    4. Evaluate and compare results
    5. Generate visualizations
    """
    print("=" * 60)
    print("TINYML HAR COMPRESSION PIPELINE")
    print("=" * 60)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  - Pruning: {'ON' if args.prune else 'OFF'} (amount: {args.prune_amount:.0%})")
    print(f"  - Quantization: {'ON' if args.quantize else 'OFF'} ({args.quantize_bits}-bit)")
    
    if args.prune:
        print(f"\n  Note: Light pruning (≤50%) can maintain or improve")
        print(f"        accuracy via regularization effect.")
    
    # === STEP 1: Data Preparation ===
    print("\n[1/5] Preparing dataset...")
    download_dataset()
    train_loader, test_loader, timesteps = prepare_data()
    
    # === STEP 2: Train FP32 Baseline ===
    print("\n[2/5] Training FP32 baseline model...")
    fp32_model = HAR_CNN()
    
    # Model statistics
    total_params = sum(p.numel() for p in fp32_model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Theoretical FP32 size: {total_params * 4 / 1024:.2f} KB")
    print(f"   Theoretical INT8 size: {total_params * 1 / 1024:.2f} KB (25% of FP32)")
    
    # Training
    train_model(fp32_model, train_loader)
    fp32_acc = evaluate(fp32_model, test_loader)
    print(f"   FP32 accuracy: {fp32_acc:.4f}")
    
    # Save baseline
    fp32_path = os.path.join(RESULTS_DIR, "model_fp32.pt")
    save_model_dense(fp32_model, fp32_path, "FP32 baseline")
    fp32_size = get_model_size(fp32_path)
    print(f"   FP32 actual size: {fp32_size / 1024:.2f} KB")
    
    # Results storage
    results = [("FP32 Baseline", fp32_acc, fp32_size, 100.0, 100.0)]
    file_paths = {"FP32 Baseline": fp32_path}
    
    # === STEP 3: Apply Compression ===
    print("\n[3/5] Applying compression...")
    model_cpu = fp32_model.cpu()
    model_cpu.eval()
    
    # Case A: Pruning only
    if args.prune and not args.quantize:
        print(f"\n  Applying pruning only ({args.prune_amount:.0%})...")
        
        # Prune model
        pruned_model, sparsity, _ = apply_pruning(model_cpu, amount=args.prune_amount)
        pruned_acc = evaluate(pruned_model, test_loader)
        acc_percentage = (pruned_acc / fp32_acc) * 100 if fp32_acc > 0 else 100
        print(f"\n   Pruned accuracy: {pruned_acc:.4f} ({acc_percentage:.1f}% of baseline)")
        
        # Save in sparse format
        pruned_sparse_path = os.path.join(RESULTS_DIR, f"model_pruned_{int(args.prune_amount*100)}_sparse.npz")
        _, _ = save_pruned_model_sparse(pruned_model, pruned_sparse_path, sparsity)
        pruned_sparse_size = get_model_size(pruned_sparse_path)
        size_percentage = (pruned_sparse_size / fp32_size) * 100 if fp32_size > 0 else 100
        
        print(f"\n   Actual sparse size: {pruned_sparse_size / 1024:.2f} KB ({size_percentage:.1f}% of baseline)")
        
        # Verify loading works
        print(f"\n   Verifying sparse model loading...")
        loaded_model = load_pruned_model_sparse(pruned_sparse_path)
        loaded_acc = evaluate(loaded_model, test_loader)
        print(f"   Loaded accuracy: {loaded_acc:.4f}")
        
        # Store results
        results.append((f"Pruned ({int(args.prune_amount*100)}%)", 
                       pruned_acc, pruned_sparse_size, acc_percentage, size_percentage))
        file_paths[f"Pruned ({int(args.prune_amount*100)}%)"] = pruned_sparse_path
    
    # Case B: Quantization only
    elif args.quantize and not args.prune:
        print(f"\n  Applying {args.quantize_bits}-bit quantization only...")
        
        # Quantize model
        quantized_model, q_weights, q_scales, theoretical_size_percentage = apply_quantization(
            model_cpu, precision_bits=args.quantize_bits
        )
        quantized_acc = evaluate(quantized_model, test_loader)
        acc_percentage = (quantized_acc / fp32_acc) * 100 if fp32_acc > 0 else 100
        print(f"\n   Quantized accuracy: {quantized_acc:.4f} ({acc_percentage:.1f}% of baseline)")
        
        # Save quantized model
        quantized_path = os.path.join(RESULTS_DIR, f"model_quantized_{args.quantize_bits}bit.npz")
        save_quantized_model(q_weights, q_scales, args.quantize_bits, quantized_path)
        quantized_size = get_model_size(quantized_path)
        size_percentage = (quantized_size / fp32_size) * 100 if fp32_size > 0 else 100
        
        print(f"   Actual quantized size: {quantized_size / 1024:.2f} KB ({size_percentage:.1f}% of baseline)")
        
        results.append((f"Quantized ({args.quantize_bits}-bit)", 
                       quantized_acc, quantized_size, acc_percentage, size_percentage))
        file_paths[f"Quantized ({args.quantize_bits}-bit)"] = quantized_path
    
    # Case C: Both pruning and quantization
    elif args.prune and args.quantize:
        print(f"\n  Applying both pruning and quantization...")
        
        # First: Prune
        print(f"    Step 1: Pruning ({args.prune_amount:.0%})...")
        pruned_model, sparsity, _ = apply_pruning(model_cpu, amount=args.prune_amount)
        pruned_acc = evaluate(pruned_model, test_loader)
        acc_after_pruning = (pruned_acc / fp32_acc) * 100 if fp32_acc > 0 else 100
        print(f"    After pruning accuracy: {pruned_acc:.4f} ({acc_after_pruning:.1f}% of baseline)")
        
        # Second: Create combined sparse+quantized format
        print(f"    Step 2: {args.quantize_bits}-bit quantization of sparse weights...")
        
        combined_data = {}
        if args.quantize_bits == 4:
            qmin, qmax = -7, 7
            bytes_per_element = 0.5
        elif args.quantize_bits == 8:
            qmin, qmax = -127, 127
            bytes_per_element = 1.0
        else:  # 16-bit
            qmin, qmax = -32767, 32767
            bytes_per_element = 2.0
        
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                if 'weight' in name and param.numel() > 0:
                    # Process non-zero weights only
                    non_zero_mask = param != 0
                    non_zero_values = param[non_zero_mask]
                    
                    if non_zero_values.numel() > 0:
                        # Quantize non-zero values
                        abs_max = non_zero_values.abs().max().item()
                        scale = abs_max / qmax if abs_max > 0 else 1.0
                        quantized = torch.round(non_zero_values / scale).clamp(qmin, qmax)
                        
                        # Store sparse-quantized representation
                        non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False)
                        combined_data[f'{name}_values'] = quantized.detach().cpu().numpy().astype(
                            np.int8 if args.quantize_bits == 8 else np.int16
                        )
                        combined_data[f'{name}_indices'] = non_zero_indices.detach().cpu().numpy().astype(np.uint16)
                        combined_data[f'{name}_shape'] = np.array(param.shape, dtype=np.uint16)
                        combined_data[f'{name}_scale'] = np.float32(scale)
                    else:
                        # Handle all-zeros weight tensor
                        combined_data[f'{name}_values'] = np.array([], dtype=np.int8 if args.quantize_bits == 8 else np.int16)
                        combined_data[f'{name}_indices'] = np.array([], dtype=np.uint16).reshape(0, 2)
                        combined_data[f'{name}_shape'] = np.array(param.shape, dtype=np.uint16)
                        combined_data[f'{name}_scale'] = np.float32(1.0)
                else:
                    # Store biases normally (FP32)
                    combined_data[name] = param.detach().cpu().numpy().astype(np.float32)
        
        # Add metadata
        combined_data['sparsity'] = np.float32(sparsity)
        combined_data['quantization_bits'] = np.int32(args.quantize_bits)
        
        # Save combined model
        compressed_path = os.path.join(RESULTS_DIR, 
            f"model_pruned{int(args.prune_amount*100)}_quantized{args.quantize_bits}bit_sparse.npz")
        np.savez_compressed(compressed_path, **combined_data)
        
        compressed_size = get_model_size(compressed_path)
        size_percentage = (compressed_size / fp32_size) * 100 if fp32_size > 0 else 100
        
        print(f"\n   Combined model:")
        print(f"   Size: {compressed_size/1024:.2f} KB ({size_percentage:.1f}% of baseline)")
        
        # Store results (use pruned accuracy as reference)
        results.append((f"Pruned+Quantized", 
                       pruned_acc, compressed_size, acc_after_pruning, size_percentage))
        file_paths["Pruned+Quantized"] = compressed_path
    
    else:
        print("\n  No compression selected. Only baseline FP32 model created.")
    
    # === STEP 4: Display Results ===
    print("\n[4/5] Compression Results (vs FP32 Baseline)")
    print("=" * 75)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'% of Baseline':<12} {'Size (KB)':<12} {'% of Baseline':<12}")
    print("-" * 75)
    
    for name, acc, size, acc_pct, size_pct in results:
        size_kb = size / 1024
        print(f"{name:<25} {acc:<12.4f} {acc_pct:<12.1f}% {size_kb:<12.2f} {size_pct:<12.1f}%")
    
    # === STEP 5: Create Visualizations ===
    print("\n[5/5] Creating visualizations...")
    
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        model_names = [r[0] for r in results]
        accuracies = [r[1] for r in results]
        size_percentages = [r[4] for r in results]
        acc_percentages = [r[3] for r in results]
        
        # Accuracy plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        x_pos = range(len(model_names))
        
        bars1 = ax1.bar(x_pos, accuracies, color=colors, edgecolor='black')
        ax1.set_ylim(0, max(accuracies) * 1.15)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add accuracy values and percentages
        for i, (bar, acc, acc_pct) in enumerate(zip(bars1, accuracies, acc_percentages)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
            if i > 0:  # Percentage for compressed models
                ax1.text(bar.get_x() + bar.get_width()/2, 0.02,
                        f'{acc_pct:.1f}%', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='darkblue')
        
        # Size plot (percentage of baseline)
        bars2 = ax2.bar(x_pos, size_percentages, color=colors, edgecolor='black')
        ax2.set_title('Model Size (% of FP32)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Size (% of FP32)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add size percentages and values
        for i, (bar, size_pct) in enumerate(zip(bars2, size_percentages)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{size_pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            if i < len(results):
                size_kb = results[i][2] / 1024
                ax2.text(bar.get_x() + bar.get_width()/2, 5,
                        f'{size_kb:.1f} KB', ha='center', va='bottom', fontsize=9)
        
        # Highlight baseline (100%)
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(len(model_names)-0.5, 102, 'FP32 Baseline (100%)', 
                ha='right', va='bottom', color='red', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = "compression_results"
        if args.prune:
            plot_filename += f"_prune{int(args.prune_amount*100)}"
        if args.quantize:
            plot_filename += f"_quant{args.quantize_bits}"
        plot_filename += ".png"
        
        plot_path = os.path.join(RESULTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n   Plot saved: {plot_path}")
    else:
        print("   Skipping plot - only baseline model available.")
        plot_path = None
    
    # === Save Summary File ===
    summary_path = os.path.join(RESULTS_DIR, "compression_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("HAR MODEL COMPRESSION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Pruning: {'Enabled' if args.prune else 'Disabled'}")
        if args.prune:
            f.write(f" (amount: {args.prune_amount:.0%})\n")
        else:
            f.write("\n")
        
        f.write(f"  Quantization: {'Enabled' if args.quantize else 'Disabled'}")
        if args.quantize:
            f.write(f" ({args.quantize_bits}-bit)\n")
        else:
            f.write("\n")
        
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  FP32 baseline size: {fp32_size/1024:.2f} KB\n")
        f.write(f"  FP32 baseline accuracy: {fp32_acc:.4f}\n\n")
        
        f.write("Results (vs FP32 baseline):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':<12} {'%':<12} {'Size (KB)':<12} {'%':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, acc, size, acc_pct, size_pct in results:
            size_kb = size / 1024
            f.write(f"{name:<25} {acc:<12.4f} {acc_pct:<12.1f}% {size_kb:<12.2f} {size_pct:<12.1f}%\n")
    
    print(f"\n   Summary saved: {summary_path}")
    
    # === File Listing ===
    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)
    
    for name, path in file_paths.items():
        print(f"  - {name}: {path}")
    
    if plot_path:
        print(f"  - Visualization: {plot_path}")
    print(f"  - Summary: {summary_path}")
    print("=" * 60)
    
    # === Quick Analysis ===
    print("\nQUICK ANALYSIS:")
    for name, acc, size, acc_pct, size_pct in results[1:]:  # Skip baseline
        print(f"  {name}:")
        print(f"    Accuracy: {acc_pct:.1f}% of baseline")
        print(f"    Size: {size_pct:.1f}% of baseline")
        if size_pct > 0:
            print(f"    Efficiency: {acc_pct/size_pct:.2f} (acc%/size%)")
    
    print("\n[COMPLETE] Pipeline finished successfully.")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyML HAR Model Compression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline only
  python %(prog)s
  
  # 30% pruning
  python %(prog)s --prune
  
  # 50% pruning  
  python %(prog)s --prune --prune_amount 0.5
  
  # INT8 quantization
  python %(prog)s --quantize
  
  # INT4 quantization
  python %(prog)s --quantize --quantize_bits 4
  
  # Combined: 30% pruning + INT8
  python %(prog)s --prune --quantize
  
  # Combined: 50% pruning + INT4
  python %(prog)s --prune --prune_amount 0.5 --quantize --quantize_bits 4
        """
    )
    
    parser.add_argument("--prune", action="store_true",
                       help="Apply magnitude-based pruning")
    
    parser.add_argument("--prune_amount", type=float, default=0.3,
                       help="Pruning sparsity (0.0 to 1.0, default: 0.3)")
    
    parser.add_argument("--quantize", action="store_true",
                       help="Apply quantization")
    
    parser.add_argument("--quantize_bits", type=int, default=8,
                       choices=[4, 8, 16],
                       help="Quantization precision (default: 8-bit)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.prune_amount < 0 or args.prune_amount > 1:
        print("Error: prune_amount must be between 0.0 and 1.0")
        exit(1)
    
    if not args.prune and not args.quantize:
        print("No compression selected. Creating baseline FP32 model only.")
        print("Use --prune for pruning or --quantize for quantization.")
    
    # Run pipeline
    main(args)