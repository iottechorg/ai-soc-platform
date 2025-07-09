#!/bin/bash

set -e

echo "🔨 Building SOC Platform Docker Images (Fixed Dependencies)..."

# Check if requirements files exist
if [ ! -f "requirements-base.txt" ]; then
    echo "❌ requirements-base.txt not found. Creating from template..."
    cat > requirements-base.txt << EOF
kafka-python==2.0.2
clickhouse-driver==0.2.6
redis==4.6.0
pyyaml==6.0
python-dotenv==1.0.0
requests==2.31.0
prometheus-client==0.17.1
schedule==1.2.0
numpy==1.24.3
EOF
fi

if [ ! -f "requirements-ml.txt" ]; then
    echo "❌ requirements-ml.txt not found. Creating from template..."
    cat > requirements-ml.txt << EOF
scikit-learn==1.3.0
pandas==2.0.3
EOF
fi

if [ ! -f "requirements-rl.txt" ]; then
    echo "❌ requirements-rl.txt not found. Creating from template..."
    cat > requirements-rl.txt << EOF
gymnasium==0.26.3
stable-baselines3==1.8.0
torch==2.0.1
EOF
fi

if [ ! -f "requirements-dashboard.txt" ]; then
    echo "❌ requirements-dashboard.txt not found. Creating from template..."
    cat > requirements-dashboard.txt << EOF
streamlit==1.28.0
plotly==5.15.0
pandas==2.0.3
EOF
fi

# Build images with better error handling
echo "Building Data Generator..."
docker build -f Dockerfile.data-generator -t soc-platform/data-generator:latest . || {
    echo "❌ Failed to build data-generator"
    exit 1
}

echo "Building ML Pipeline..."
docker build -f Dockerfile.ml-pipeline -t soc-platform/ml-pipeline:latest . || {
    echo "❌ Failed to build ml-pipeline"
    exit 1
}

echo "Building RL Orchestrator..."
docker build -f Dockerfile.orchestrator -t soc-platform/orchestrator:latest . || {
    echo "❌ Failed to build orchestrator"
    exit 1
}

echo "Building Scoring Engine..."
docker build -f Dockerfile.scoring -t soc-platform/scoring-engine:latest . || {
    echo "❌ Failed to build scoring-engine"
    exit 1
}

echo "Building Alerting Service..."
docker build -f Dockerfile.alerting -t soc-platform/alerting:latest . || {
    echo "❌ Failed to build alerting"
    exit 1
}

echo "Building Dashboard..."
docker build -f Dockerfile.dashboard -t soc-platform/dashboard:latest . || {
    echo "❌ Failed to build dashboard"
    exit 1
}

echo "✅ All images built successfully!"

echo ""
echo "🔍 Verifying images..."
docker images | grep soc-platform

echo ""
echo "🚀 Ready to deploy with:"
echo "  docker-compose up -d"