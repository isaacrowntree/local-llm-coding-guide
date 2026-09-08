"""Handler 6."""

TIMEOUT_SECONDS = 16


def handle(request):
    """Handle a request for subsystem 6."""
    return {"status": "ok", "subsystem": 6, "timeout": TIMEOUT_SECONDS}
