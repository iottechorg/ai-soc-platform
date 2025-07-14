# Data Generator Dockerfile
FROM soc-platform-base:latest

# Set the working directory
WORKDIR /app

# Copy application code
COPY ./services/shared ./shared
COPY ./config ./config

# 3. Copy the service's own source code
COPY ./services/data_generator/src .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Fixed start command
CMD ["/usr/local/bin/wait-for-it.sh", "kafka:29092", "--timeout=60", "--", "python", "run_data_generator.py"]
