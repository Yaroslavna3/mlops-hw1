import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_pipeline.data import create_dataloader
from ml_pipeline.model import RegressionModel, create_model


def load_model(model_path: str | Path) -> RegressionModel:
    """Load a trained regression model."""
    model = create_model()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(
    model: RegressionModel,
    dataloader: DataLoader,
) -> float:
    """Calculate mean squared error on a dataset."""
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    samples_count = 0

    with torch.no_grad():
        for features, targets in dataloader:
            predictions = model(features)
            total_loss += criterion(predictions, targets).item()
            samples_count += len(features)

    return total_loss / samples_count


def save_metrics(mse: float, output_path: str | Path) -> None:
    """Save evaluation metrics as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"mse": mse}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    dataloader = create_dataloader("data/raw.csv", batch_size=4)
    model = load_model("artifacts/model.pt")
    mse = evaluate_model(model, dataloader)
    save_metrics(mse, "reports/metrics.json")

    print(f"MSE: {mse:.6f}")
    print("Metrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()
