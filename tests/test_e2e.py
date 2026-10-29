import json
from pathlib import Path

import pytest

from ml_pipeline.data import create_dataloader
from ml_pipeline.evaluation import evaluate_model, load_model, save_metrics
from ml_pipeline.training import save_model, train_model


@pytest.mark.e2e
def test_complete_pipeline_creates_model_and_metrics(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    model_path = tmp_path / "artifacts" / "model.pt"
    metrics_path = tmp_path / "reports" / "metrics.json"
    csv_path.write_text(
        "x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n",
        encoding="utf-8",
    )

    dataloader = create_dataloader(csv_path, batch_size=3, shuffle=True)
    model, _ = train_model(dataloader, epochs=250)
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    mse = evaluate_model(loaded_model, dataloader)
    save_metrics(mse, metrics_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["mse"] < 0.01
