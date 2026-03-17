# ─────────────────────────────────────────────────────────────────────────────
# WA Solar Grid Optimizer — Phase 4 serving image
# Python 3.12 matches pyproject.toml target-version
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install runtime deps + serving stack into an isolated prefix
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir \
        -r requirements.txt \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "httpx>=0.27"


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="WA Solar Grid Optimizer API" \
      org.opencontainers.image.description="FastAPI DNI inference service — Perth, WA" \
      org.opencontainers.image.version="1.0.0"

# Non-root user
RUN useradd --create-home --shell /bin/bash solar

WORKDIR /app

# Packages from builder
COPY --from=builder /install /usr/local

# Application source
COPY app/main.py ./main.py
COPY src/ ./src/

# Artefact directories (populated at runtime via bind-mount or COPY)
RUN mkdir -p /app/artifacts && chown -R solar:solar /app

USER solar

# ── Environment defaults ───────────────────────────────────────────────────────
ENV MODEL_DIR=/app/artifacts \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info

EXPOSE 8000

CMD ["sh", "-c", \
     "uvicorn main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers ${WORKERS} \
        --log-level ${LOG_LEVEL} \
        --no-access-log"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Stage 3: lean production image ───────────────────────────────────────────────
FROM python:3.12-slim AS production
LABEL org.opencontainers.image.version="1.0.0"

RUN useradd --create-home --shell /bin/bash solar
WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements-serve.txt

# Copy app and pre-exported artefacts directly into the image
COPY app/main.py ./main.py
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/scaler_Daylight_Optimized.pkl ./artifacts/scaler_Daylight_Optimized.pkl

RUN chown -R solar:solar /app
USER solar

ENV MODEL_DIR=/app/artifacts \
    PORT=8000

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
