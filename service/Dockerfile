# CPU-friendly base for Cloud Run / generic servers
FROM python:3.10-slim

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY service /app/service

# Model directory (copy your TorchScript into this path before building, or mount at runtime)
RUN mkdir -p /app/runs_resnet50
# If bundling model at build time, uncomment the next line after placing the model file:
COPY service/runs_resnet50/resnet50_pneumonia_traced.pt /app/runs_resnet50/resnet50_pneumonia_traced.pt

ENV MODEL_PATH=/app/runs_resnet50/resnet50_pneumonia_traced.pt
ENV IMG_SIZE=224
ENV THRESHOLD=0.20
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "service.model_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
