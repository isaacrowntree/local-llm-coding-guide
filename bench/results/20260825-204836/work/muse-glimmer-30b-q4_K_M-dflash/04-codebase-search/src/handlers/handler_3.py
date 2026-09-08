"""Handler 3."""

TIMEOUT_SECONDS = 13


def handle(request):
    """Handle a request for subsystem 3."""
    return {"status": "ok", "subsystem": 3, "timeout": TIMEOUT_SECONDS}
