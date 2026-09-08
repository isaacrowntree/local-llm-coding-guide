"""Handler 12."""

TIMEOUT_SECONDS = 22


def handle(request):
    """Handle a request for subsystem 12."""
    return {"status": "ok", "subsystem": 12, "timeout": TIMEOUT_SECONDS}
