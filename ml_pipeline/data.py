import csv
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset for a simple one-dimensional regression problem."""

    def __init__(self, csv_path: str | Path) -> None:
        features: list[list[float]] = []
        targets: list[list[float]] = []

        with Path(csv_path).open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)

            for row in reader:
                features.append([float(row["x"])])
                targets.append([float(row["y"])])

        if not features:
            raise ValueError("dataset must not be empty")

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]


def create_dataloader(
    csv_path: str | Path,
    batch_size: int = 4,
    shuffle: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a DataLoader for the regression dataset."""
    dataset = RegressionDataset(csv_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
