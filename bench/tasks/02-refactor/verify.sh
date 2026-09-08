#!/bin/bash
# Verify task 02 in $1. Exit 0 = pass.
W="$1"
cd "$W" || exit 1

if grep -rn "fetch_data" --include="*.py" . >/dev/null 2>&1; then
  echo "FAIL: 'fetch_data' still present:"
  grep -rn "fetch_data" --include="*.py" . | head -5
  exit 1
fi

grep -q "def load_records" store/source.py 2>/dev/null || {
  echo "FAIL: load_records not defined in store/source.py"
  exit 1
}

python3 -m unittest discover -s tests -t . -q >/dev/null 2>&1 || {
  echo "FAIL: unittest suite failing after rename"
  exit 1
}
exit 0
