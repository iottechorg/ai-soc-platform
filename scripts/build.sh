#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "🔨 Building all SOC Platform services via Docker Compose..."
echo "   (This will build the base image first, then all dependent services)"

# The 'docker compose build' command reads the docker-compose.yml file
# and builds all services with a 'build' section, in the correct order.
docker compose build

echo "✅ All images built successfully!"