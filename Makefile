# Makefile (Updated)
.PHONY: help build deploy start stop restart logs clean health status fix-deps

# Default target
help:
	@echo "SOC Platform Management Commands"
	@echo "================================="
	@echo "fix-deps  - Create compatible requirements files"
	@echo "build     - Build all Docker images"
	@echo "deploy    - Deploy the complete platform"
	@echo "start     - Start all services"
	@echo "stop      - Stop all services"
	@echo "restart   - Restart all services"
	@echo "logs      - Show logs (make logs SERVICE=service-name)"
	@echo "clean     - Clean up containers and images"
	@echo "health    - Run health checks"
	@echo "status    - Show service status"

fix-deps:
	@echo "🔧 Creating compatible requirements files..."
	@bash scripts/build.sh

build: fix-deps
	@bash scripts/build.sh

deploy: build
	@bash scripts/deploy.sh

start:
	@docker-compose up -d

stop:
	@docker-compose down

restart: stop start

logs:
	@bash scripts/logs.sh $(SERVICE)

clean:
	@bash scripts/clean.sh

health:
	@bash scripts/health-check.sh

status:
	@docker-compose ps
