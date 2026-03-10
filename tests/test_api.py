"""
Tests for app/main.py — FastAPI endpoints (Phase 4)

Follows project conventions:
- Class-based test organisation
- Monkeypatching instead of decorators where possible
- tmp_path for file isolation
- No internet / no real model needed
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_model_state(monkeypatch):
    """
    Inject a mock sklearn estimator and scaler into app state.
    Prevents any disk/MLflow access during tests.
    """
    import app.main as api

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([550.0])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 4))

    monkeypatch.setattr(api, "_load_model", lambda: None)
    api.state.model = mock_model
    api.state.scaler = mock_scaler
    api.state.model_meta = {"source": "mock"}
    api.state.load_time = 1_700_000_000.0

    return mock_model, mock_scaler


@pytest.fixture()
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


# ── Valid payload (matches SolarFeatures schema) ───────────────────────────────
VALID_OBS = {
    "temperature_2m": 28.5,
    "cloud_cover": 10.0,
    "hour": 12,
    "month": 6,
}


# ── /health ────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_scaler_loaded_true(self, client):
        assert client.get("/health").json()["scaler_loaded"] is True

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_version_present(self, client):
        assert "version" in client.get("/health").json()

    def test_model_source_reported(self, client):
        assert client.get("/health").json()["model_source"] == "mock"


# ── /predict ───────────────────────────────────────────────────────────────────
class TestPredict:
    def test_valid_payload_returns_200(self, client):
        assert client.post("/predict", json=VALID_OBS).status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/predict", json=VALID_OBS).json()
        assert "direct_normal_irradiance_wm2" in data
        assert "model_source" in data
        assert "latency_ms" in data

    def test_prediction_value_matches_mock(self, client):
        data = client.post("/predict", json=VALID_OBS).json()
        assert data["direct_normal_irradiance_wm2"] == pytest.approx(550.0, abs=1e-3)

    def test_model_predict_called_once(self, client, mock_model_state):
        mock_model, _ = mock_model_state
        client.post("/predict", json=VALID_OBS)
        mock_model.predict.assert_called_once()

    def test_scaler_transform_called(self, client, mock_model_state):
        _, mock_scaler = mock_model_state
        client.post("/predict", json=VALID_OBS)
        mock_scaler.transform.assert_called_once()

    def test_cloud_cover_above_100_rejected(self, client):
        bad = {**VALID_OBS, "cloud_cover": 110.0}
        assert client.post("/predict", json=bad).status_code == 422

    def test_cloud_cover_below_0_rejected(self, client):
        bad = {**VALID_OBS, "cloud_cover": -5.0}
        assert client.post("/predict", json=bad).status_code == 422

    def test_hour_above_23_rejected(self, client):
        bad = {**VALID_OBS, "hour": 25}
        assert client.post("/predict", json=bad).status_code == 422

    def test_month_above_12_rejected(self, client):
        bad = {**VALID_OBS, "month": 13}
        assert client.post("/predict", json=bad).status_code == 422

    def test_temperature_out_of_wa_range_rejected(self, client):
        bad = {**VALID_OBS, "temperature_2m": 100.0}
        assert client.post("/predict", json=bad).status_code == 422

    def test_missing_field_rejected(self, client):
        bad = {k: v for k, v in VALID_OBS.items() if k != "cloud_cover"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_negative_raw_prediction_clipped_to_zero(self, client, mock_model_state):
        mock_model, _ = mock_model_state
        mock_model.predict.return_value = np.array([-200.0])
        data = client.post("/predict", json=VALID_OBS).json()
        assert data["direct_normal_irradiance_wm2"] >= 0.0


# ── /predict/batch ─────────────────────────────────────────────────────────────
class TestPredictBatch:
    def test_single_obs_returns_200(self, client):
        r = client.post("/predict/batch", json={"observations": [VALID_OBS]})
        assert r.status_code == 200

    def test_count_matches_input(self, client, mock_model_state):
        mock_model, mock_scaler = mock_model_state
        mock_model.predict.return_value = np.array([400.0, 600.0, 300.0])
        mock_scaler.transform.return_value = np.zeros((3, 4))
        r = client.post("/predict/batch", json={"observations": [VALID_OBS] * 3})
        assert r.json()["count"] == 3

    def test_result_list_length_matches_input(self, client, mock_model_state):
        mock_model, mock_scaler = mock_model_state
        mock_model.predict.return_value = np.array([400.0, 600.0])
        mock_scaler.transform.return_value = np.zeros((2, 4))
        r = client.post("/predict/batch", json={"observations": [VALID_OBS] * 2})
        assert len(r.json()["direct_normal_irradiance_wm2"]) == 2

    def test_empty_observations_rejected(self, client):
        assert (
            client.post("/predict/batch", json={"observations": []}).status_code == 422
        )

    def test_response_has_required_fields(self, client):
        data = client.post("/predict/batch", json={"observations": [VALID_OBS]}).json()
        for key in (
            "direct_normal_irradiance_wm2",
            "count",
            "model_source",
            "latency_ms",
        ):
            assert key in data


# ── /model/reload ──────────────────────────────────────────────────────────────
class TestModelReload:
    def test_returns_200(self, client):
        assert client.post("/model/reload").status_code == 200

    def test_reloaded_flag_true(self, client):
        assert client.post("/model/reload").json()["reloaded"] is True


# ── 503 when model absent ──────────────────────────────────────────────────────
class TestNoModel:
    def test_predict_returns_503_when_no_model(self, client):
        import app.main as api

        original = api.state.model
        api.state.model = None
        try:
            assert client.post("/predict", json=VALID_OBS).status_code == 503
        finally:
            api.state.model = original

    def test_health_status_degraded_when_no_model(self, client):
        import app.main as api

        original = api.state.model
        api.state.model = None
        try:
            assert client.get("/health").json()["status"] == "degraded"
        finally:
            api.state.model = original
