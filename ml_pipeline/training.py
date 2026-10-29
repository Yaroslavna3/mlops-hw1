from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_pipeline.data import create_dataloader
from ml_pipeline.model import RegressionModel, create_model


def train_model(
    dataloader: DataLoader,
    epochs: int = 300,
    learning_rate: float = 0.01,
) -> tuple[RegressionModel, list[float]]:
    """Train the regression model and return its loss history."""
    model = create_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        samples_count = 0

        for features, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(features)
            samples_count += len(features)

        losses.append(epoch_loss / samples_count)

    return model, losses


def save_model(model: RegressionModel, output_path: str | Path) -> None:
    """Save the trained model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    dataloader = create_dataloader(
        "data/raw.csv",
        batch_size=4,
        shuffle=True,
    )
    model, losses = train_model(dataloader)
    save_model(model, "artifacts/model.pt")

    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print("Model saved to artifacts/model.pt")


if __name__ == "__main__":
    main()
