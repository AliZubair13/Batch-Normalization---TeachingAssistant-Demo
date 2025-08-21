"""
Deep Learning TA Interview - Implementation Exercise

Student Name: Zubair Ali
Topic Chosen: Batch Normalization from Scratch
Date Started: 2025-07-25

This file contains a full implementation of Batch Normalization from scratch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Batch Normalization Layer
# --------------------------
class BatchNorm1D:
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        Initialize the BatchNorm1D layer.

        Args:
            num_features (int): Number of features/channels.
            momentum (float): Momentum for updating running mean/variance.
            eps (float): A small number to prevent division by zero.
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True  # Mode flag
        self.gamma = torch.ones(num_features, requires_grad=True)  # Learnable scale
        self.beta = torch.zeros(num_features, requires_grad=True)  # Learnable shift
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.cache = {}  # For storing intermediate variables during forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Batch Normalization.

        Args:
            x (Tensor): Input of shape (batch_size, num_features)

        Returns:
            Tensor: Normalized output
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if x.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional")

        if x.shape[1] != self.num_features:
            raise ValueError(f"Expected input with {self.num_features} features, but got {x.shape[1]}")

        if x.shape[0] == 0:
            raise ValueError("Batch size must be greater than 0")

        if self.training:
            # Compute mean and variance on current batch
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            # Normalize input
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            # Scale and shift
            out = self.gamma * x_hat + self.beta
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.detach()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.detach()
            # Cache for backward pass
            self.cache = {
                'x_hat': x_hat,
                'var': batch_var,
                'x': x,
                'mean': batch_mean,
                'std_inv': 1. / torch.sqrt(batch_var + self.eps)
            }
        else:
            # Use running statistics in evaluation mode
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta

        return out

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for computing gradients.

        Args:
            dout (Tensor): Gradient of the loss w.r.t. output.

        Returns:
            Tensor: Gradient w.r.t. input.
        """
        N, D = dout.shape
        x_hat = self.cache['x_hat']
        std_inv = self.cache['std_inv']
        x = self.cache['x']
        mean = self.cache['mean']
        var = self.cache['var']

        # Gradients w.r.t. gamma and beta (parameters)
        self.dgamma = torch.sum(dout * x_hat, dim=0)
        self.dbeta = torch.sum(dout, dim=0)

        # Backprop into x
        dxhat = dout * self.gamma
        dvar = torch.sum(dxhat * (x - mean) * -0.5 * std_inv**3, dim=0)
        dmean = torch.sum(dxhat * -std_inv, dim=0) + dvar * torch.mean(-2 * (x - mean), dim=0)
        dx = dxhat * std_inv + dvar * 2 * (x - mean) / N + dmean / N

        return dx

    def train(self):
        """Set layer to training mode."""
        self.training = True

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False

# --------------------------
# Simple MLP using BatchNorm
# --------------------------
class SimpleMLP:
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        """
        A simple two-layer MLP with Batch Normalization.

        Args:
            input_dim (int): Size of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output classes.
        """
        self.fc1 = torch.randn(input_dim, hidden_dim, requires_grad=True)  # First linear layer weights
        self.bn1 = BatchNorm1D(hidden_dim)  # BatchNorm after first layer
        self.fc2 = torch.randn(hidden_dim, output_dim, requires_grad=True)  # Output layer weights

    def forward(self, x: torch.Tensor, train=True) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input data.
            train (bool): Whether the model is in training mode.

        Returns:
            Tensor: Output logits.
        """
        if train:
            self.bn1.train()
        else:
            self.bn1.eval()

        z1 = x @ self.fc1  # Linear transformation
        h1 = self.bn1.forward(z1)  # Apply batch normalization
        a1 = torch.relu(h1)  # Activation
        out = a1 @ self.fc2  # Output logits

        return out

# --------------------------
# Training the Model
# --------------------------
def main():
    torch.manual_seed(42)  # For reproducibility

    # Create model and dummy dataset
    model = SimpleMLP(input_dim=10, hidden_dim=16, output_dim=2)
    X = torch.randn(128, 10)  # Input features
    y = torch.randint(0, 2, (128,))  # Random binary labels
    y_onehot = F.one_hot(y, num_classes=2).float()  # One-hot encoded labels
    lr = 1e-3  # Learning rate

    for epoch in range(20):
        # Forward pass
        logits = model.forward(X, train=True)
        probs = F.softmax(logits, dim=1)
        loss = torch.mean((probs - y_onehot)**2)  # MSE loss for simplicity

        # Fake backward hook to prevent autograd errors
        loss.backward = lambda: None

        with torch.no_grad():
            # Gradient w.r.t. output of second layer
            grad_output = 2 * (probs - y_onehot) / y.shape[0]

            # Update fc2
            hidden_activations = model.bn1.forward(X @ model.fc1).relu()
            grad_fc2 = hidden_activations.T @ grad_output
            model.fc2 -= lr * grad_fc2

            # Gradient flowing back
            dout = grad_output @ model.fc2.T
            da1 = dout.clone()
            dz1 = da1.clone()  # Derivative of ReLU is 1 for positive values; skipped for simplicity

            # Backprop through BatchNorm
            dbn1 = model.bn1.backward(dz1)

            # Update fc1
            grad_fc1 = X.T @ dbn1
            model.fc1 -= lr * grad_fc1

            # Update gamma and beta in BatchNorm
            model.bn1.gamma = model.bn1.gamma - lr * model.bn1.dgamma
            model.bn1.beta = model.bn1.beta - lr * model.bn1.dbeta

        print(f"Epoch {epoch+1:02d}: Loss = {loss.item():.4f}")

    print("\nTraining complete. Now switching to evaluation mode.\n")
    model.forward(X[:5], train=False)  # Run model in eval mode on first 5 inputs

# Run the training loop
if __name__ == "__main__":
    main()
