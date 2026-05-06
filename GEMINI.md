# PulmoLens Project Instructions

## Backend Infrastructure & Deployment
- **Google Cloud Run**: The backend API is deployed on Google Cloud Run.
- **Port Configuration**: The backend FastAPI/Uvicorn server must dynamically bind to the `PORT` environment variable provided by Cloud Run. Do not hardcode ports in the `Dockerfile` CMD or the application's startup script.
- **Memory Requirements**: The backend relies on PyTorch for ML inference and a RAG stack. Cloud Run's default memory limits (e.g., 512 MiB) are insufficient and will cause Out of Memory (OOM) errors during the container startup when loading model weights. The deployment must specify a minimum of `4Gi` memory allocation (e.g., `--memory=4Gi`).