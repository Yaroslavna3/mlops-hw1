from pathlib import Path

import pytest
import torch

from ml_pipeline.data import RegressionDataset, create_dataloader


@pytest.mark.unit
def test_regression_dataset_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,3\n2,5\n", encoding="utf-8")

    dataset = RegressionDataset(csv_path)

    assert len(dataset) == 2
    features, target = dataset[0]
    assert torch.equal(features, torch.tensor([1.0]))
    assert torch.equal(target, torch.tensor([3.0]))


@pytest.mark.unit
def test_create_dataloader_builds_batches(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,3\n2,5\n3,7\n", encoding="utf-8")

    dataloader = create_dataloader(csv_path, batch_size=2)
    batches = list(dataloader)

    assert len(batches) == 2
    assert batches[0][0].shape == (2, 1)
    assert batches[0][1].shape == (2, 1)
