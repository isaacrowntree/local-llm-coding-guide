"""Handler 10."""

TIMEOUT_SECONDS = 20


def handle(request):
    """Handle a request for subsystem 10."""
    return {"status": "ok", "subsystem": 10, "timeout": TIMEOUT_SECONDS}
