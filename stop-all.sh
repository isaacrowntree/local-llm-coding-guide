#!/bin/bash
# Stop all local LLM services
# Usage: ./stop-all.sh

echo "Stopping llama-server..."
pkill -f llama-server 2>/dev/null && echo "  stopped" || echo "  not running"

echo "Stopping LiteLLM..."
pkill -f litellm 2>/dev/null && echo "  stopped" || echo "  not running"

echo "Stopping cloudflared..."
pkill -f cloudflared 2>/dev/null && echo "  stopped" || echo "  not running"

echo "Done."
