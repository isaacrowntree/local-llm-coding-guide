"""Handler 8."""

TIMEOUT_SECONDS = 18


def handle(request):
    """Handle a request for subsystem 8."""
    return {"status": "ok", "subsystem": 8, "timeout": TIMEOUT_SECONDS}
