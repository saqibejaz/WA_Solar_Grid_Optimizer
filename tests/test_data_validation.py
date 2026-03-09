"""
Tests for data_validation.py — validate_perth_data()

We write temp CSVs from our fixtures so no real data file is needed.
"""

import os

import pandas as pd

from data_validation import validate_perth_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_temp_csv(df: pd.DataFrame, tmp_path) -> str:
    path = os.path.join(tmp_path, "test_solar.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidatePerthData:
    def test_clean_data_passes(self, sample_solar_df, tmp_path):
        """Clean data with no night-sun should return True."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        result = validate_perth_data(file_path=path)
        assert result is True

    def test_dirty_data_fails(self, dirty_solar_df, tmp_path):
        """Data with irradiance at midnight should return False."""
        path = write_temp_csv(dirty_solar_df, tmp_path)
        result = validate_perth_data(file_path=path)
        assert result is False

    def test_missing_values_detected(self, sample_solar_df, tmp_path):
        """Validation should run even when NaNs are present (no crash)."""
        df = sample_solar_df.copy()
        df.loc[0, "cloud_cover"] = None
        path = write_temp_csv(df, tmp_path)
        result = validate_perth_data(file_path=path)
        assert isinstance(result, bool)

    def test_returns_bool(self, sample_solar_df, tmp_path):
        """validate_perth_data() must always return a boolean."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        result = validate_perth_data(file_path=path)
        assert isinstance(result, bool)
