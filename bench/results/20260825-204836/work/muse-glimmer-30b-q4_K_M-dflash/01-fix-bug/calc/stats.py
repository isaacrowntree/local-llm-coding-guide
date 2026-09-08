"""Small statistics helpers."""


def mean(values):
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)


def median(values):
    if not values:
        raise ValueError("median() requires at least one value")
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid]


def variance(values):
    if len(values) < 2:
        raise ValueError("variance() requires at least two values")
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)
