def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    if not values:
        raise ValueError("values must not be empty")

    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Return the median of a non-empty list."""
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2

    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]

    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def main() -> None:
    numbers = [1.0, 2.0, 3.0]
    print(f"Mean: {mean(numbers)}")
    print(f"Median: {median(numbers)}")


if __name__ == "__main__":
    main()