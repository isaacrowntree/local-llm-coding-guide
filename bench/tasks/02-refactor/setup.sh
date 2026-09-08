#!/bin/bash
# Build the fixture for task 02 in $1
set -e
W="$1"
mkdir -p "$W/store" "$W/tests"

cat > "$W/store/__init__.py" <<'PY'
PY

cat > "$W/store/source.py" <<'PY'
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
PY

cat > "$W/store/report.py" <<'PY'
from store.source import fetch_data


def names(limit=None):
    return [r["name"] for r in fetch_data(limit)]


def count(limit=None):
    return len(fetch_data(limit))
PY

cat > "$W/store/cli.py" <<'PY'
import sys

from store.source import fetch_data
from store.report import names


def main(argv):
    if "--names" in argv:
        print(",".join(names()))
    else:
        print(len(fetch_data()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PY

cat > "$W/tests/__init__.py" <<'PY'
PY

cat > "$W/tests/test_store.py" <<'PY'
import unittest

from store.source import fetch_data
from store.report import names, count


class TestStore(unittest.TestCase):
    def test_fetch_all(self):
        self.assertEqual(len(fetch_data()), 3)

    def test_fetch_limit(self):
        self.assertEqual(len(fetch_data(2)), 2)

    def test_names(self):
        self.assertEqual(names(), ["alpha", "beta", "gamma"])

    def test_count(self):
        self.assertEqual(count(1), 1)


if __name__ == "__main__":
    unittest.main()
PY
