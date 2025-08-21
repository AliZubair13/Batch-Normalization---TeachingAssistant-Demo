"""
Utility functions for the TA interview implementation.

Add any helper functions, visualization utilities, or
common operations here.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_loss_curve(losses: List[float], title: str = "Training Loss"):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()


# Add more utility functions as needed for your implementation
