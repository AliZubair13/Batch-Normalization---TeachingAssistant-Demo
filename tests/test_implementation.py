import torch
import pytest
import torch.nn.functional as F
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from implementation import BatchNorm1D, SimpleMLP, main


class TestBatchNorm1D:
    """Unit tests for the BatchNorm1D class."""

    def test_forward_training(self) -> None:
        """Test output shape and zero-mean property in training mode."""
        bn = BatchNorm1D(4)
        bn.train()
        x = torch.randn(10, 4)
        out = bn.forward(x)
        assert out.shape == x.shape
        assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-1)

    def test_forward_eval(self) -> None:
        """Test forward pass behavior in evaluation mode using running statistics."""
        bn = BatchNorm1D(4)
        x = torch.randn(10, 4)
        bn.train()
        bn.forward(x)  # Update running stats
        bn.eval()
        out = bn.forward(x)
        assert out.shape == x.shape

    def test_forward_invalid_type(self) -> None:
        """Ensure that passing an invalid type raises a TypeError."""
        bn = BatchNorm1D(4)
        with pytest.raises(TypeError):
            bn.forward("invalid")  # type: ignore

    def test_forward_invalid_dimension(self) -> None:
        """Ensure that non-2D input raises a ValueError."""
        bn = BatchNorm1D(4)
        with pytest.raises(ValueError):
            bn.forward(torch.randn(4))

    def test_forward_mismatched_features(self) -> None:
        """Ensure that mismatched feature dimensions raise a ValueError."""
        bn = BatchNorm1D(4)
        with pytest.raises(ValueError):
            bn.forward(torch.randn(2, 3))

    def test_forward_empty_batch(self) -> None:
        """Ensure that an empty batch raises a ValueError."""
        bn = BatchNorm1D(4)
        with pytest.raises(ValueError):
            bn.forward(torch.empty(0, 4))

    def test_backward_shape(self) -> None:
        """Test backward pass shape and gradients for gamma and beta."""
        bn = BatchNorm1D(4)
        x = torch.randn(5, 4)
        bn.train()
        out = bn.forward(x)
        dout = torch.randn_like(out)
        dx = bn.backward(dout)
        assert dx.shape == x.shape
        assert hasattr(bn, 'dgamma')
        assert hasattr(bn, 'dbeta')

    def test_backward_without_forward(self) -> None:
        """Ensure that backward without forward raises a KeyError."""
        bn = BatchNorm1D(4)
        dout = torch.randn(5, 4)
        with pytest.raises(KeyError):
            bn.backward(dout)

    def test_train_eval_switch(self) -> None:
        """Test toggling between training and evaluation modes."""
        bn = BatchNorm1D(4)
        bn.eval()
        assert not bn.training
        bn.train()
        assert bn.training

    def test_forward_small_batch(self) -> None:
        """Test forward pass with a batch size of 1."""
        bn = BatchNorm1D(4)
        x = torch.randn(1, 4)
        out = bn.forward(x)
        assert out.shape == x.shape

    def test_zero_variance_input(self) -> None:
        """Test forward pass when input has zero variance."""
        bn = BatchNorm1D(4)
        x = torch.ones(10, 4)
        out = bn.forward(x)
        assert not torch.isnan(out).any()


class TestSimpleMLP:
    """Unit tests for the SimpleMLP class."""

    def test_forward_train(self) -> None:
        """Test output shape of the model in training mode."""
        model = SimpleMLP(input_dim=10, hidden_dim=8, output_dim=2)
        x = torch.randn(16, 10)
        out = model.forward(x, train=True)
        assert out.shape == (16, 2)

    def test_forward_eval(self) -> None:
        """Test output shape of the model in evaluation mode."""
        model = SimpleMLP(input_dim=10, hidden_dim=8, output_dim=2)
        x = torch.randn(16, 10)
        model.forward(x, train=True)
        out = model.forward(x, train=False)
        assert out.shape == (16, 2)

    def test_single_sample(self) -> None:
        """Test model forward pass with batch size of 1."""
        model = SimpleMLP()
        x = torch.randn(1, 10)
        out = model.forward(x)
        assert out.shape == (1, 2)

    def test_large_batch(self) -> None:
        """Test model forward pass with large batch size."""
        model = SimpleMLP()
        x = torch.randn(4096, 10)
        out = model.forward(x)
        assert out.shape == (4096, 2)

    def test_output_does_not_require_grad(self) -> None:
        """Ensure model output doesn't require gradients in no_grad context."""
        model = SimpleMLP()
        x = torch.randn(64, 10)
        with torch.no_grad():
            out = model.forward(x)
            assert out.requires_grad is False

    def test_parameter_updates(self) -> None:
        """Ensure model parameters are updated after simulated gradient step."""
        model = SimpleMLP()
        x = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))
        y_onehot = F.one_hot(y, num_classes=2).float()

        # Save parameter copies
        before_fc1 = model.fc1.clone()
        before_fc2 = model.fc2.clone()
        before_gamma = model.bn1.gamma.clone()
        before_beta = model.bn1.beta.clone()

        # Manual forward and backward
        logits = model.forward(x)
        probs = torch.softmax(logits, dim=1)
        grad_output = 2 * (probs - y_onehot) / y.shape[0]
        grad_fc2 = model.bn1.forward(x @ model.fc1).relu().T @ grad_output

        model.fc2 = (model.fc2 - 1e-3 * grad_fc2).detach().requires_grad_()
        dout = grad_output @ model.fc2.T
        dbn1 = model.bn1.backward(dout)
        grad_fc1 = x.T @ dbn1
        model.fc1 = (model.fc1 - 1e-3 * grad_fc1).detach().requires_grad_()
        model.bn1.gamma = (model.bn1.gamma - 1e-3 * model.bn1.dgamma).detach().requires_grad_()
        model.bn1.beta = (model.bn1.beta - 1e-3 * model.bn1.dbeta).detach().requires_grad_()

        # Ensure parameters changed
        assert not torch.equal(before_fc1, model.fc1)
        assert not torch.equal(before_fc2, model.fc2)
        assert not torch.equal(before_gamma, model.bn1.gamma)
        assert not torch.equal(before_beta, model.bn1.beta)


def test_main_runs() -> None:
    """Basic integration test to ensure main() runs without error."""
    main()
