import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables full deterministic mode (slower but reproducible)
    """
    # Set Python hash seed for deterministic string hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seeds for random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # CUDA determinism settings
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CUBLAS workspace config for deterministic operations
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # MPS (Apple Silicon) determinism
    if torch.backends.mps.is_available() and deterministic:
        # MPS doesn't have specific determinism flags yet, but seed is set
        pass

    # Enable deterministic algorithms (may fail for some operations)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            warnings.warn(
                f"Could not enable deterministic algorithms: {e}. "
                "Some operations may be non-deterministic."
            )
