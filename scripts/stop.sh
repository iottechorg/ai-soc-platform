#!/bin/bash
set -e

echo "🛑 Stopping all SOC Platform services gracefully..."

# The --timeout flag gives containers 30 seconds to shut down before being killed.
docker compose down --timeout 30

echo "✅ SOC Platform stopped."