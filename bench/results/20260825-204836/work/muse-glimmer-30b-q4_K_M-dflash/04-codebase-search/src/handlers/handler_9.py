"""Handler 9."""

TIMEOUT_SECONDS = 19


def handle(request):
    """Handle a request for subsystem 9."""
    return {"status": "ok", "subsystem": 9, "timeout": TIMEOUT_SECONDS}
