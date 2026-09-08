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
