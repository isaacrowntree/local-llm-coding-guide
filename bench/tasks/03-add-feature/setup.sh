#!/bin/bash
# Build the fixture for task 03 in $1
set -e
W="$1"
mkdir -p "$W"

cat > "$W/todo.py" <<'PY'
#!/usr/bin/env python3
"""Tiny todo CLI.

Usage:
  todo.py list
  todo.py add <text>
"""
import sys

TASKS = [
    {"id": 1, "text": "write tests", "done": True},
    {"id": 2, "text": "ship it", "done": False},
]


def cmd_list(argv):
    for t in TASKS:
        mark = "x" if t["done"] else " "
        print(f"[{mark}] {t['id']} {t['text']}")
    return 0


def cmd_add(argv):
    if not argv:
        print("add requires text", file=sys.stderr)
        return 1
    new_id = max((t["id"] for t in TASKS), default=0) + 1
    TASKS.append({"id": new_id, "text": " ".join(argv), "done": False})
    print(f"added {new_id}")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        return cmd_list(rest)
    if cmd == "add":
        return cmd_add(rest)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PY
chmod +x "$W/todo.py"

cat > "$W/README.md" <<'MD'
# todo

A tiny todo CLI. Commands: `list`, `add <text>`.
MD
