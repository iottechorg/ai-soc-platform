FROM soc-platform-base:latest

WORKDIR /app
COPY ./services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./services/shared ./shared
COPY ./config ./config

# 3. Copy the service's own source code
COPY ./services/dashboard/src .


# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Fixed start command
CMD ["/usr/local/bin/wait-for-it.sh", "clickhouse:8123", "--timeout=60", "--", \
     "python", "-m", "streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]