FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_PATH=/models/model.json
WORKDIR /app
COPY requirements/serve.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home learner
COPY capstones /app/capstones
USER learner
EXPOSE 8000
CMD ["uvicorn", "capstones.rf_detection.app:app", "--host", "0.0.0.0", "--port", "8000"]
