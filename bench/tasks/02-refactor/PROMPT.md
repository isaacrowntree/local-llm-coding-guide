Rename the function `fetch_data` to `load_records` throughout this repository.

Update every definition, every import, and every call site — including the tests
in tests/, which should be updated to use the new name. The public behaviour must
not change.

When you are done, no occurrence of the string `fetch_data` should remain anywhere
in the repository, and `python3 -m unittest discover -s tests -t .` must pass.
