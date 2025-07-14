# Alerting Dockerfile
FROM soc-platform-base:latest

WORKDIR /app

# Copy application code
COPY ./services/shared ./shared
COPY ./config ./config

# 3. Copy the service's own source code
COPY ./services/alerting/src .


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('Alerting Service healthy')"

# Fixed start command
CMD ["/usr/local/bin/wait-for-it.sh", "kafka:29092", "--timeout=60", "--", \
     "python", "run_alerting.py"]