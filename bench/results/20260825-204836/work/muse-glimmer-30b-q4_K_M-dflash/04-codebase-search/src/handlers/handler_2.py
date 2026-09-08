"""Handler 2."""

TIMEOUT_SECONDS = 12


def handle(request):
    """Handle a request for subsystem 2."""
    return {"status": "ok", "subsystem": 2, "timeout": TIMEOUT_SECONDS}
