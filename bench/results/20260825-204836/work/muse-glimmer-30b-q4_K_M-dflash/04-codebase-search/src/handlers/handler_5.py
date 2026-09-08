"""Handler 5."""

TIMEOUT_SECONDS = 15


def handle(request):
    """Handle a request for subsystem 5."""
    return {"status": "ok", "subsystem": 5, "timeout": TIMEOUT_SECONDS}
