import pytest

from main import mean, median


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_empty() -> None:
    with pytest.raises(ValueError):
        mean([])


def test_median_odd() -> None:
    assert median([3.0, 1.0, 2.0]) == 2.0


def test_median_even() -> None:
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_median_empty() -> None:
    with pytest.raises(ValueError, match="median values must not be empty"):
        median([])
