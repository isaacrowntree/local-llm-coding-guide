"""Handler 11."""

TIMEOUT_SECONDS = 21


def handle(request):
    """Handle a request for subsystem 11."""
    return {"status": "ok", "subsystem": 11, "timeout": TIMEOUT_SECONDS}
