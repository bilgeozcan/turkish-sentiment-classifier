FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
    && useradd --create-home --uid 10001 appuser

COPY src/ ./src/
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.sentiment.app:app", "--host", "0.0.0.0", "--port", "8000"]
