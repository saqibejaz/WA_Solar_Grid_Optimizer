"""
WA Solar Grid Optimizer — Phase 4: FastAPI Model Serving
Serves direct_normal_irradiance predictions using the trained RandomForest.
Features: temperature_2m, cloud_cover, hour, month
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.info("MLflow not available — joblib-only mode.")

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from scalar_fastapi import get_scalar_api_reference

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("wa_solar.api")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts"))
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{Path(__file__).parent.parent / 'mlflow.db'}",
)

# Matches preprocessing.py: features = ["temperature_2m", "cloud_cover", "hour", "month"]
FEATURE_COLUMNS = ["temperature_2m", "cloud_cover", "hour", "month"]


# ── App state ─────────────────────────────────────────────────────────────────
class AppState:
    model: Any = None
    scaler: Any = None
    model_meta: dict = {}
    load_time: float = 0.0


state = AppState()


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model() -> None:
    """
    Load order:
      1. MLflow registry    → models:/wa_solar_irradiance/Production  (dev only)
      2. Latest MLflow run  → most recent run logged by train_mlflow.py (dev only)
      3. Latest *.joblib    → MODEL_DIR (production image fallback)
      4. Latest scaler_*.pkl → artifacts/ (always)
    """
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # 1. MLflow registry
        try:
            model_uri = "models:/wa_solar_irradiance/Production"
            state.model = mlflow.sklearn.load_model(model_uri)
            state.model_meta = {"source": "mlflow_registry", "uri": model_uri}
            log.info("✅ Model loaded from MLflow registry: %s", model_uri)
        except Exception as exc:
            log.warning(
                "MLflow registry unavailable (%s), trying latest MLflow run …", exc
            )

            # 2. Latest MLflow run
            try:
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(
                    experiment_ids=["0"],
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/random_forest_model"
                    state.model = mlflow.sklearn.load_model(model_uri)
                    state.model_meta = {"source": "mlflow_run", "run_id": run_id}
                    log.info("✅ Model loaded from MLflow run: %s", run_id)
                else:
                    raise ValueError("No MLflow runs found")
            except Exception as exc2:
                log.warning("MLflow run load failed (%s), trying local .joblib …", exc2)
                _load_joblib()
    else:
        # Production image — mlflow not installed, go straight to joblib
        log.info("MLflow not available — loading from joblib artefact.")
        _load_joblib()

    # Scaler — always loaded from artifacts/ regardless of model source
    scaler_candidates = sorted(
        MODEL_DIR.glob("scaler_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if scaler_candidates:
        state.scaler = joblib.load(scaler_candidates[0])
        log.info("✅ Scaler loaded from: %s", scaler_candidates[0])
    else:
        log.warning("No scaler_*.pkl found — raw features will be passed to model.")
        state.scaler = None

    state.load_time = time.time()


def _load_joblib() -> None:
    """Fallback: load the latest .joblib from MODEL_DIR."""
    candidates = sorted(
        MODEL_DIR.glob("*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        log.error("No model found anywhere. Run src/train_mlflow.py first.")
        state.model = None
        return

    model_path = candidates[0]
    state.model = joblib.load(model_path)
    state.model_meta = {"source": "joblib", "path": str(model_path)}
    log.info("✅ Model loaded from: %s", model_path)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🔆 WA Solar Grid Optimizer API starting up …")
    _load_model()
    yield
    log.info("🌙 Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="WA Solar Grid Optimizer",
    description=(
        "Predicts direct normal irradiance (W/m²) for Perth, WA. "
        "Features: temperature_2m, cloud_cover, hour, month."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # disable default Swagger UI
    redoc_url=None,  # disable default ReDoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class SolarFeatures(BaseModel):
    """
    Mirrors preprocessing.py feature list exactly:
    ["temperature_2m", "cloud_cover", "hour", "month"]
    """

    temperature_2m: float = Field(
        ..., description="Air temperature at 2 m (°C)", examples=[22.5]
    )
    cloud_cover: float = Field(
        ..., ge=0, le=100, description="Total cloud cover (%)", examples=[15.0]
    )
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)", examples=[12])
    month: int = Field(
        ..., ge=1, le=12, description="Month of year (1–12)", examples=[6]
    )

    @field_validator("temperature_2m")
    @classmethod
    def _temp_plausible(cls, v: float) -> float:
        if not (-10.0 <= v <= 55.0):
            raise ValueError("temperature_2m outside plausible WA range (-10 to 55 °C)")
        return v


class BatchRequest(BaseModel):
    observations: list[SolarFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    direct_normal_irradiance_wm2: float = Field(..., description="Predicted DNI (W/m²)")
    model_source: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    direct_normal_irradiance_wm2: list[float]
    count: int
    model_source: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    model_source: str | None
    uptime_seconds: float | None
    version: str


# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_dataframe(obs: list[SolarFeatures]) -> pd.DataFrame:
    rows = [o.model_dump() for o in obs]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _predict(df: pd.DataFrame) -> np.ndarray:
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health for details.",
        )
    X = state.scaler.transform(df) if state.scaler is not None else df.values
    preds = state.model.predict(X)
    return np.clip(preds, 0, None)  # DNI is non-negative


# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def _timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    uptime = round(time.time() - state.load_time, 1) if state.load_time else None
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        scaler_loaded=state.scaler is not None,
        model_source=state.model_meta.get("source"),
        uptime_seconds=uptime,
        version=app.version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Single-observation DNI prediction",
)
def predict(payload: SolarFeatures):
    t0 = time.perf_counter()
    df = _to_dataframe([payload])
    result = _predict(df)
    latency = (time.perf_counter() - t0) * 1000
    log.info("predict | dni=%.1f W/m² | %.2f ms", result[0], latency)
    return PredictionResponse(
        direct_normal_irradiance_wm2=float(result[0]),
        model_source=state.model_meta.get("source", "unknown"),
        latency_ms=round(latency, 3),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
    summary="Batch DNI predictions (max 1 000 rows)",
)
def predict_batch(payload: BatchRequest):
    t0 = time.perf_counter()
    df = _to_dataframe(payload.observations)
    results = _predict(df)
    latency = (time.perf_counter() - t0) * 1000
    log.info("predict_batch | n=%d | %.2f ms", len(results), latency)
    return BatchPredictionResponse(
        direct_normal_irradiance_wm2=[float(v) for v in results],
        count=len(results),
        model_source=state.model_meta.get("source", "unknown"),
        latency_ms=round(latency, 3),
    )


@app.post(
    "/model/reload",
    tags=["ops"],
    summary="Hot-reload model without container restart",
)
def reload_model():
    log.info("Hot-reload requested …")
    _load_model()
    return {"reloaded": True, "model_source": state.model_meta.get("source")}


@app.get("/docs", include_in_schema=False)
def scalar_docs():
    """Scalar API docs — served locally, no CDN needed."""
    return get_scalar_api_reference(
        openapi_url="/openapi.json",
        title="WA Solar Grid Optimizer",
    )


@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("""
<!doctype html>
<html>
  <head>
    <title>WA Solar Grid Optimizer</title>
    <meta charset="utf-8"/>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #0a0a0a; color: #e5e5e5;
        min-height: 100vh; display: flex;
        align-items: center; justify-content: center;
      }
      .card {
        border: 1px solid #2a2a2a; border-radius: 12px;
        padding: 48px 56px; max-width: 480px; width: 100%; background: #111;
      }
      .badge {
        display: inline-block; background: #1a3a1a; color: #4ade80;
        font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 4px 10px;
        border-radius: 4px; margin-bottom: 20px;
      }
      h1 { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
      p  { font-size: 14px; color: #888; line-height: 1.6; margin-bottom: 32px; }
      .links { display: flex; gap: 12px; }
      a { text-decoration: none; font-size: 13px; font-weight: 500; padding: 9px 20px; border-radius: 6px; }
      .primary { background: #f5f5f5; color: #0a0a0a; }
      .secondary { border: 1px solid #2a2a2a; color: #aaa; }
      .primary:hover { background: #fff; }
      .secondary:hover { border-color: #444; color: #e5e5e5; }
      .meta { margin-top: 36px; padding-top: 24px; border-top: 1px solid #1e1e1e; display: flex; gap: 24px; }
      .meta span { font-size: 12px; color: #555; }
      .meta strong { color: #888; display: block; font-size: 13px; }
    </style>
  </head>
  <body>
    <div class="card">
      <div class="badge">&#9679; Live</div>
      <h1>WA Solar Grid Optimizer</h1>
      <p>Direct normal irradiance prediction for Perth, WA.<br/>RandomForest · MLflow · FastAPI</p>
      <div class="links">
        <a class="primary" href="/docs">API Docs</a>
        <a class="secondary" href="/health">Health</a>
      </div>
      <div class="meta">
        <div><strong>v1.0.0</strong><span>Version</span></div>
        <div><strong>POST /predict</strong><span>Inference</span></div>
        <div><strong>Perth, WA</strong><span>Location</span></div>
      </div>
    </div>
  </body>
</html>
""")
