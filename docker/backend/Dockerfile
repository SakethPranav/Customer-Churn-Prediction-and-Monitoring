FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so this slow layer is cached across code changes.
COPY docker/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at runtime.
COPY src/__init__.py src/__init__.py
COPY src/api src/api
COPY src/common src/common
COPY src/monitoring src/monitoring
COPY monitoring monitoring

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
