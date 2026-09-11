# OceanEmbed Phase 5 — Hardened Production Serving Dockerfile
# Python 3.12 Slim CPU Base Image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    API_ENV=demo \
    ENABLE_DOCS=false \
    OCEAN_DATA_MODE=synthetic \
    OCEANEMBED_CHECKPOINT=checkpoints/oceanembed_v2.pt \
    ARGO_DATA_DIR=data/raw/argo \
    SATELLITE_DATA_DIR=data/raw/satellite

WORKDIR /app

# Install minimal system utilities and create dedicated non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 appgroup \
    && useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Copy dependency specifications and install dependencies
COPY requirements.txt .

# Install CPU-only PyTorch wheel and core requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration, and artifacts
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY checkpoints/ /app/checkpoints/
COPY data/ /app/data/
COPY dashboard/ /app/dashboard/

# Assign safe permissions to non-root appuser
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI HTTP serving port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint with graceful signal handling
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

