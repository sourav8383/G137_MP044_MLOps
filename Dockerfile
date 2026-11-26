FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY params.yaml .
COPY src/ src/
COPY models/model.joblib models/model.joblib

RUN useradd --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "mp044.api:app", "--host", "0.0.0.0", "--port", "8000"]
