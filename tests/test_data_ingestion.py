"""
Tests for data_ingestion.py — fetch_wa_solar_data()

We mock the requests.get() call so tests never need internet access.
This is standard practice — you test YOUR code, not the API.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd

from data_ingestion import fetch_wa_solar_data

# ---------------------------------------------------------------------------
# Fake API response — mimics what Open-Meteo actually returns
# ---------------------------------------------------------------------------

MOCK_API_RESPONSE = {
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
        "direct_normal_irradiance": [0.0, 0.0, 0.0],
        "cloud_cover": [10.0, 20.0, 30.0],
        "temperature_2m": [18.0, 17.5, 17.0],
    }
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchWaSolarData:
    @patch("data_ingestion.requests.get")
    def test_returns_dataframe_on_success(self, mock_get, tmp_path):
        """Should return a DataFrame when the API call succeeds."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_API_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output = os.path.join(tmp_path, "perth_solar_raw.csv")
        result = fetch_wa_solar_data(output_path=output)
        assert isinstance(result, pd.DataFrame)

    @patch("data_ingestion.requests.get")
    def test_dataframe_has_correct_columns(self, mock_get, tmp_path):
        """Returned DataFrame should contain the expected columns."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_API_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output = os.path.join(tmp_path, "perth_solar_raw.csv")
        result = fetch_wa_solar_data(output_path=output)
        expected_cols = {
            "time",
            "direct_normal_irradiance",
            "cloud_cover",
            "temperature_2m",
        }
        assert expected_cols.issubset(set(result.columns))

    @patch("data_ingestion.requests.get")
    def test_returns_none_on_api_failure(self, mock_get, tmp_path):
        """Should return None gracefully when the API call fails."""
        mock_get.side_effect = Exception("Network error")

        output = os.path.join(tmp_path, "perth_solar_raw.csv")
        result = fetch_wa_solar_data(output_path=output)
        assert result is None

    @patch("data_ingestion.requests.get")
    def test_time_column_is_datetime(self, mock_get, tmp_path):
        """The time column should be parsed as datetime, not a string."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_API_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output = os.path.join(tmp_path, "perth_solar_raw.csv")
        result = fetch_wa_solar_data(output_path=output)
        assert pd.api.types.is_datetime64_any_dtype(result["time"])
