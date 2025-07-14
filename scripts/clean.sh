#!/bin/bash
set -e

echo "🧹 Cleaning up SOC Platform..."

# Stop and remove all containers, networks, and volumes defined in docker-compose.
echo "   -> Stopping and removing containers and networks..."
docker compose down -v --remove-orphans

# Find all images with the 'soc-platform/' prefix and forcefully remove them.
echo "   -> Removing built Docker images..."
docker images "soc-platform/*" -q | xargs -r docker rmi -f
docker images "soc-platform-base" -q | xargs -r docker rmi -f


echo "✅ Cleanup complete!"