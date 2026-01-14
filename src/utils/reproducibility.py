import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.rand