# OceanEmbed Phase 5 — Production Serving Dockerfile
# Python 3.12 Slim CPU Base Image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    OCEAN_DATA_MODE=synthetic \
    OCEANEMBED_CHECKPOINT=checkpoints/oceanembed_v2.pt \
    ARGO_DATA_DIR=data/raw/argo \
    SATELLITE_DATA_DIR=data/raw/satellite

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install dependencies
COPY requirements.txt .

# Install CPU-only PyTorch wheel and core requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration, and trained checkpoints
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY checkpoints/ /app/checkpoints/
COPY data/ /app/data/
COPY dashboard/ /app/dashboard/

# Expose FastAPI HTTP serving port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
