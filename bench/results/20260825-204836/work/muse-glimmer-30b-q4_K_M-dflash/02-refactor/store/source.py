"""Record source."""

RECORDS = [
    {"id": 1, "name": "alpha"},
    {"id": 2, "name": "beta"},
    {"id": 3, "name": "gamma"},
]


def fetch_data(limit=None):
    """Return records, optionally limited."""
    if limit is None:
        return list(RECORDS)
    return list(RECORDS[:limit])
