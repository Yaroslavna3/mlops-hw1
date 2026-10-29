import torch
from torch import Tensor, nn


class RegressionBlock(nn.Module):
    """Linear layer followed by a ReLU activation."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, features: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self.activation(self.linear(features))


class RegressionModel(nn.Module):
    """Small neural network for one-dimensional regression."""

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.block = RegressionBlock(input_size=1, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self.output(self.block(features))


def create_model(seed: int = 42) -> RegressionModel:
    """Create a reproducibly initialized regression model."""
    torch.manual_seed(seed)
    return RegressionModel()
