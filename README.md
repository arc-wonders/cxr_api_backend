# CXR Pneumonia API (FastAPI + Docker)

This is a production-ready **backend** for your pneumonia detection model.
It serves a **/predict** endpoint that takes a chest X-ray image and returns
`probability`, `label`, `threshold`, `latency_ms`, and `device`.

> **Note:** This repo ships with an empty `service/runs_resnet50/` folder.
> Copy your TorchScript file here as:
>
> `service/runs_resnet50/resnet50_pneumonia_traced.pt`

## Quick Start (Docker)

```bash
# 1) Build the image
docker build -t cxr-api:latest -f service/Dockerfile .

# 2) Copy your model into the image context (one time)
#    Make sure you have: service/runs_resnet50/resnet50_pneumonia_traced.pt

# 3) Run the container
docker run --rm -p 8080:8080 cxr-api:latest

# 4) Test
python scripts/test_request.py  # or use curl as below
```

### Test with curl
```bash
curl -F "file=@sample_cxr.jpg" "http://localhost:8080/predict?threshold=0.25"
```

### Test with Python
```bash
python scripts/test_request.py --image path/to/cxr.jpg --threshold 0.25
```

## Endpoints

- `GET /healthz` → `{ "status": "ok", "device": "cpu|cuda" }`
- `GET /version` → build and model info
- `POST /predict` (multipart form: `file`) → JSON with prediction

## Environment Variables

- `MODEL_PATH` (default: `/app/runs_resnet50/resnet50_pneumonia_traced.pt`)
- `IMG_SIZE` (default: `224`)
- `THRESHOLD` (default: `0.20`)

## Notes

- The Docker image uses a CPU base image for portability (works on Cloud Run).
- For GPU deployment, use an NVIDIA CUDA base and add `--gpus all` at runtime.
- This backend is **for research/demo only** and **not** a medical device.
