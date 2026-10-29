import pytest
import torch

from ml_pipeline.model import RegressionBlock, create_model


@pytest.mark.unit
def test_regression_block_preserves_batch_dimension() -> None:
    block = RegressionBlock(input_size=1, hidden_size=4)
    features = torch.tensor([[-1.0], [2.0], [3.0]])

    output = block(features)

    assert output.shape == (3, 4)
    assert torch.all(output >= 0)


@pytest.mark.unit
def test_regression_model_returns_one_prediction_per_sample() -> None:
    model = create_model(seed=7)
    features = torch.tensor([[1.0], [2.0]])

    output = model(features)

    assert output.shape == (2, 1)
