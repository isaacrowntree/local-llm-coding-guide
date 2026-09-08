from store.source import fetch_data


def names(limit=None):
    return [r["name"] for r in fetch_data(limit)]


def count(limit=None):
    return len(fetch_data(limit))
