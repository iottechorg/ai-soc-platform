#!/bin/bash

# This script shows logs for a specific service or all services.
# Usage: ./scripts/logs.sh [service-name]

SERVICE=${1:-}

if [ -z "$SERVICE" ]; then
    echo "📋 Tailing logs for all services... (Press Ctrl+C to stop)"
    docker compose logs -f --tail="50"
else
    echo "📋 Tailing logs for '$SERVICE'... (Press Ctrl+C to stop)"
    docker compose logs -f --tail="50" "$SERVICE"
fi