# PulmoLens Enterprise Backend - Optimized AI Runtime
FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system utilities for OpenCV (Headless)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install AI dependencies (Cached)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy standalone backend logic
COPY api.py .
COPY model.py .
COPY ingest_docs.py .
# DATA directories used by RAG
COPY data data

# Note: The PyTorch model weights (.pth) are NOT baked into this image.
# Weights are pulled at runtime from Azure Blob Storage via MODEL_SAS_URL.

# Cloud Run provides the PORT environment variable
ENV PORT=8080
EXPOSE $PORT

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
