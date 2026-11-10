FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow==3.18.0 cloudpickle==3.1.2

# Data directory owned by the non-root user, so the named volume is writable.
RUN useradd --create-home appuser && mkdir /mlflow && chown appuser /mlflow
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')" || exit 1

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/mlartifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]
