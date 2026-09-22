# Lesson 59: Docker for ML deployment

A container packages code and dependencies into a reproducible runtime.

## Minimal Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Learn
- image vs container
- build context
- layers
- ports
- volumes
- environment variables
- CPU vs GPU containers

## Exercise
Containerize the FastAPI model service and verify the same prediction locally and inside the container.
