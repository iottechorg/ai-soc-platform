#!/bin/bash

echo "🏥 SOC Platform Health Check"
echo "================================"

echo "Container Status:"
docker compose ps
echo ""

echo "Core Service Health:"

# Check Kafka
echo -n "Kafka: "
if docker compose exec -T kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check ClickHouse
echo -n "ClickHouse: "
# Using the HTTP ping endpoint
if curl -s http://localhost:8124/ping | grep -q "Ok."; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check Dashboard (if it's running)
# We check if the container is running before trying to curl it.
if [ "$( docker container inspect -f '{{.State.Status}}' soc-dashboard 2>/dev/null )" = "running" ]; then
    echo -n "Dashboard: "
    if curl -s -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        echo "✅ Healthy"
    else
        echo "❌ Unhealthy"
    fi
else
    echo "Dashboard: ⚪ Not running"
fi

echo ""
echo "For more details, use 'make logs'"