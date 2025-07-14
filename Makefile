# Makefile (Updated)
.PHONY: help build build-base start stop restart logs clean health status 

# Default target
help:
	@echo "SOC Platform Management Commands"
	@echo "================================="
	@echo "build     - Build all Docker images"
	@echo "start     - Start all services"
	@echo "stop      - Stop all services"
	@echo "restart   - Restart all services"
	@echo "logs      - Show logs (make logs SERVICE=service-name)"
	@echo "clean     - Clean up containers and images"
	@echo "health    - Run health checks"
	@echo "status    - Show service status"

build-base:
	@echo "Building base image..."
	@docker-compose build base

build: build-base
	@echo "Building all services..."
	@bash scripts/build.sh

start: build
	@echo "Starting all services..."
	@docker-compose up -d

stop:
	@echo "Stopping all services..."
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