
# scripts/health-check.sh
#!/bin/bash

echo "🏥 SOC Platform Health Check"
echo "================================"

# Check container status
echo "Container Status:"
docker-compose ps

echo ""
echo "Service Health Checks:"

# Check Kafka
echo -n "Kafka: "
if docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check ClickHouse
echo -n "ClickHouse: "
if curl -s http://localhost:8123/ping >/dev/null; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check Redis
echo -n "Redis: "
if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check Dashboard
echo -n "Dashboard: "
if curl -s http://localhost:8501/_stcore/health >/dev/null; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check Grafana
echo -n "Grafana: "
if curl -s http://localhost:3000/api/health >/dev/null; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check Prometheus
echo -n "Prometheus: "
if curl -s http://localhost:9090/-/healthy >/dev/null; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
