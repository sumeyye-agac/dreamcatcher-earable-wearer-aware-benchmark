import torch
import csv
import os
from typing import Dict


def count_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_cpu_latency(model: torch.nn.Module, input_shape: tuple, n_warmup: int = 10, n_runs: int = 100) -> float:
    """
    Measure CPU inference latency in milliseconds.
    
    Args:
        model: PyTorch model
        input_shape: Input shape tuple (batch_size, channels, height, width)
        n_warmup: Number of warmup runs
        n_runs: Number of runs for averaging
        
    Returns:
        Average latency in milliseconds
    """
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    # Measure
    import time
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds
    
    return sum(times) / len(times)


def append_to_leaderboard(csv_path: str, row: Dict):
    """
    Append a row to the leaderboard CSV file.
    Creates the file with headers if it doesn't exist.
    """
    file_exists = os.path.exists(csv_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Define fieldnames based on the row structure
    fieldnames = [
        "run_name", "task", "model", "teacher", "seed", "epochs", "batch_size", "lr",
        "sr", "n_mels", "rnn_hidden", "rnn_layers", "cbam_reduction", "cbam_sa_kernel",
        "alpha", "tau", "best_val_f1", "test_acc", "test_f1", "params", "cpu_latency_ms"
    ]
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
