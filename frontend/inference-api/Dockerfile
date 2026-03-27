FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY inference-api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy server and (optionally) your model if you commit it
COPY inference-api/server.py ./server.py
COPY inference-api/models ./models

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
