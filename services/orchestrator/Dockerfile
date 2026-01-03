# Orchestartor Dockerfile
FROM soc-platform-base:latest

WORKDIR /app

# Install dependencies
COPY ./services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY ./services/shared ./shared
COPY ./config ./config

# 3. Copy the service's own source code
COPY ./services/orchestrator/src .


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "print('RL Orchestrator healthy')"

# Fixed start command
CMD ["/usr/local/bin/wait-for-it.sh", "ml-pipeline:8000", "--timeout=120", "--", \
     "python", "run_orchestrator.py"]
