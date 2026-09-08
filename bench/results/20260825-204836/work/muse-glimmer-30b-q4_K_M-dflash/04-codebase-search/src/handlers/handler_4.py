"""Handler 4."""

TIMEOUT_SECONDS = 14


def handle(request):
    """Handle a request for subsystem 4."""
    return {"status": "ok", "subsystem": 4, "timeout": TIMEOUT_SECONDS}
