#!/bin/bash
# Verify task 01 in $1. Exit 0 = pass.
W="$1"
cd "$W" || exit 1

# Tests must not have been modified
if [ -f .test_checksum ]; then
  want=$(cat .test_checksum)
  got=$(shasum -a 256 tests/test_stats.py 2>/dev/null | awk '{print $1}')
  if [ "$want" != "$got" ]; then
    echo "FAIL: tests/test_stats.py was modified (model edited the tests)"
    exit 1
  fi
fi

python3 -m unittest discover -s tests -t . -q >/dev/null 2>&1 || {
  echo "FAIL: unittest suite still failing"
  exit 1
}
exit 0
