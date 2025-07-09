
# scripts/clean.sh
#!/bin/bash

echo "🧹 Cleaning up SOC Platform..."

# Stop and remove containers
docker-compose down -v --remove-orphans

# Remove images
docker images "soc-platform/*" -q | xargs -r docker rmi -f

# Clean up volumes (uncomment if you want to remove all data)
# docker volume prune -f

echo "✅ Cleanup completed!"