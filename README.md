# MLOps Homeworks

Учебный проект по MLOps с управлением окружением через uv, Gitflow,
версионированием данных и воспроизводимым ML-пайплайном через DVC.

## ML-пайплайн

Проект обучает небольшую PyTorch-модель на синтетических данных линейной
регрессии `y = 2x + 1`.

Компоненты:

- `RegressionDataset` и PyTorch `DataLoader`;
- нейросетевой блок `RegressionBlock`;
- модель `RegressionModel`;
- обучение и сохранение модели;
- оценка модели и сохранение MSE.

## Установка

```powershell
uv sync
uv run dvc pull
```

Локальный DVC remote настроен на каталог вне Git-репозитория. На другом
компьютере путь к remote необходимо настроить заново.

## DVC

```powershell
uv run dvc repro
uv run dvc dag
uv run dvc status
uv run dvc push
```

Пайплайн состоит из двух стадий:

1. `train` создаёт `artifacts/model.pt`.
2. `evaluate` создаёт `reports/metrics.json`.

## Тесты

```powershell
uv run pytest
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m e2e
```

## Проверки качества

```powershell
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Описание Gitflow для предыдущего задания находится в
[`hw2/README.md`](hw2/README.md).
