"""Handler 7."""

TIMEOUT_SECONDS = 17


def handle(request):
    """Handle a request for subsystem 7."""
    return {"status": "ok", "subsystem": 7, "timeout": TIMEOUT_SECONDS}
