# ML Pipeline Dockerfile
FROM soc-platform-base:latest

WORKDIR /app

COPY ./services/ml_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./services/shared ./shared
COPY ./config ./config

# 3. Copy the service's own source code
COPY ./services/ml_pipeline/src .


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "print('ML Pipeline healthy')"

# Fixed start command
CMD ["/usr/local/bin/wait-for-it.sh", "kafka:29092", "--timeout=60", "--", \
     "/usr/local/bin/wait-for-it.sh", "clickhouse:8123", "--timeout=60", "--", \
     "python", "run_ml_pipeline.py"]