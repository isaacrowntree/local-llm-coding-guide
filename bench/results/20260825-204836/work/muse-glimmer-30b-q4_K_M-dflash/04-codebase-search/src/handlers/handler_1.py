"""Handler 1."""

TIMEOUT_SECONDS = 11


def handle(request):
    """Handle a request for subsystem 1."""
    return {"status": "ok", "subsystem": 1, "timeout": TIMEOUT_SECONDS}
