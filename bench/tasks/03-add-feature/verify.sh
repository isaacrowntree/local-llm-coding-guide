#!/bin/bash
# Verify task 03 in $1. Exit 0 = pass.
W="$1"
cd "$W" || exit 1

out=$(python3 todo.py list --json 2>/dev/null) || {
  echo "FAIL: 'todo.py list --json' exited non-zero"
  exit 1
}

echo "$out" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception as e:
    print("FAIL: stdout is not valid JSON: %s" % e); sys.exit(1)
if not isinstance(data, list):
    print("FAIL: JSON output is not an array"); sys.exit(1)
if len(data) != 2:
    print("FAIL: expected 2 tasks, got %d" % len(data)); sys.exit(1)
want = [{"id": 1, "text": "write tests", "done": True},
        {"id": 2, "text": "ship it", "done": False}]
for got, exp in zip(data, want):
    if not isinstance(got, dict):
        print("FAIL: element is not an object"); sys.exit(1)
    for k, v in exp.items():
        if k not in got:
            print("FAIL: missing key %r" % k); sys.exit(1)
        if got[k] != v:
            print("FAIL: %r = %r, expected %r" % (k, got[k], v)); sys.exit(1)
    if not isinstance(got["done"], bool):
        print("FAIL: done is not a boolean"); sys.exit(1)
sys.exit(0)
' || exit 1

# Plain list output must be unchanged
plain=$(python3 todo.py list 2>/dev/null)
expected='[x] 1 write tests
[ ] 2 ship it'
if [ "$plain" != "$expected" ]; then
  echo "FAIL: plain 'list' output changed"
  exit 1
fi

grep -qi -- "--json" README.md 2>/dev/null || {
  echo "FAIL: README.md does not document --json"
  exit 1
}
exit 0
