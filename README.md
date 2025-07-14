# SOC Platform Deployment Guide

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd ai-soc-platform
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Deploy Platform**
   ```bash
   make start
   ```

3. **Access Services**
   - Dashboard: http://localhost:8501

## Management Commands

- `make build` - Build all images
- `make start` - Build all images and start services
- `make stop` - Stop services
- `make logs` - View logs
- `make health` - Health check
- `make clean` - Cleanup

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8501, 3000, 9090, 8123, 9092 are available
2. **Memory issues**: Ensure at least 12GB RAM available
3. **Disk space**: Ensure at least 50GB free space

### Checking Logs
```bash
# All services
make logs

# Specific service
make logs SERVICE=ml-pipeline

# Follow logs
docker-compose logs -f data-generator
```

### Health Checks
```bash
make health
```

## Configuration

### Environment Variables
Edit `.env` file for:
- Slack webhook URL
- Email settings
- Database passwords
- Alert thresholds

### Service Configuration
Edit `config/settings.yaml` for:
- ML model parameters
- Scoring thresholds
- Kafka topics
- Database settings

## Scaling

### Horizontal Scaling
```bash
# Scale data generators
docker-compose up -d --scale data-generator=3

# Scale scoring engines
docker-compose up -d --scale scoring-engine=2
```

### Resource Limits
Adjust in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 2G
```

## Monitoring

- **ClickHouse**: http://localhost:8124
- **Streamlit**: http://localhost:8501


## Security

1. Change default passwords in `.env`
2. Use proper TLS certificates in production
3. Restrict network access
4. Regular security updates

## Medium Access
[Building the Future of Cybersecurity: An AI-Driven SOC Platform with Reinforcement Learning](https://akpolatcem.medium.com/building-the-future-of-cybersecurity-an-ai-driven-soc-platform-with-reinforcement-learning-a542820d9db8)
