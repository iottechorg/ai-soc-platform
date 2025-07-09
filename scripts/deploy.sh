# scripts/deploy.sh
#!/bin/bash

set -e

echo "🚀 Deploying SOC Platform..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create it from .env.example"
    exit 1
fi

# Create necessary directories
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p config/clickhouse
mkdir -p logs

# Start infrastructure services first
echo "Starting infrastructure services..."
docker-compose up -d zookeeper kafka clickhouse redis

# Wait for infrastructure to be ready
echo "Waiting for infrastructure services to be healthy..."
sleep 30

# Start application services
echo "Starting application services..."
docker-compose up -d data-generator ml-pipeline scoring-engine alerting

# Wait for application services
echo "Waiting for application services..."
sleep 20

# Start intelligence layer
echo "Starting intelligence layer..."
docker-compose up -d orchestrator dashboard

# Start monitoring
echo "Starting monitoring services..."
docker-compose up -d prometheus grafana

echo "✅ SOC Platform deployed successfully!"
echo ""
echo "🌐 Access URLs:"
echo "  Dashboard:   http://localhost:8501"
echo "  Grafana:     http://localhost:3000 (admin/admin123)"
echo "  Prometheus:  http://localhost:9090"
echo "  ClickHouse:  http://localhost:8123"
echo ""
echo "📊 Check service status:"
echo "  docker-compose ps"
echo ""
echo "📋 View logs:"
echo "  docker-compose logs -f [service-name]"