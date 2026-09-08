#!/bin/bash
# Build the fixture for task 04 in $1 — a haystack with one needle
set -e
W="$1"
mkdir -p "$W/src/handlers" "$W/src/config" "$W/src/util" "$W/docs"

for i in $(seq 1 12); do
  cat > "$W/src/handlers/handler_$i.py" <<PY
"""Handler $i."""

TIMEOUT_SECONDS = $((i + 10))


def handle(request):
    """Handle a request for subsystem $i."""
    return {"status": "ok", "subsystem": $i, "timeout": TIMEOUT_SECONDS}
PY
done

for i in $(seq 1 8); do
  cat > "$W/src/util/util_$i.py" <<PY
"""Utility $i."""


def transform_$i(value):
    return value * $i
PY
done

for i in $(seq 1 6); do
  cat > "$W/src/config/settings_$i.py" <<PY
"""Settings group $i."""

RETRY_LIMIT = $i
POOL_SIZE = $((i * 4))
PY
done

# The needle — buried, and referenced indirectly from a doc
cat > "$W/src/config/settings_4.py" <<'PY'
"""Settings group 4."""

RETRY_LIMIT = 4
POOL_SIZE = 16

# Tuned during the 2026 incident review; do not change without sign-off.
SHARD_REBALANCE_THRESHOLD = 0.734
PY

cat > "$W/docs/architecture.md" <<'MD'
# Architecture

Sharding is rebalanced when load exceeds the configured rebalance threshold.
The threshold lives with the settings modules under `src/config/`.
MD

for i in $(seq 1 5); do
  cat > "$W/docs/note_$i.md" <<MD
# Note $i

Operational note $i. Nothing to see here.
MD
done
