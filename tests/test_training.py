from pathlib import Path

import pytest
import torch

from ml_pipeline.data import create_dataloader
from ml_pipeline.evaluation import evaluate_model, load_model
from ml_pipeline.training import save_model, train_model


def write_regression_csv(path: Path) -> None:
    path.write_text(
        "x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n",
        encoding="utf-8",
    )


@pytest.mark.integration
def test_training_reduces_loss(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    write_regression_csv(csv_path)
    dataloader = create_dataloader(csv_path, batch_size=3)

    model, losses = train_model(dataloader, epochs=200)
    mse = evaluate_model(model, dataloader)

    assert losses[-1] < losses[0]
    assert mse < 0.01


@pytest.mark.integration
def test_saved_model_preserves_predictions(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.pt"
    write_regression_csv(csv_path)
    dataloader = create_dataloader(csv_path, batch_size=3)
    model, _ = train_model(dataloader, epochs=100)
    features = torch.tensor([[2.0], [4.0]])

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    with torch.no_grad():
        assert torch.allclose(model(features), loaded_model(features))
