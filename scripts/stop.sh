

# scripts/stop.sh
#!/bin/bash

echo "🛑 Stopping SOC Platform..."

# Graceful shutdown
docker-compose down --timeout 30

echo "✅ SOC Platform stopped successfully!"