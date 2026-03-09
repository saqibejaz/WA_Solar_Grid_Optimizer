"""
Tests for preprocessing.py — prepare_data()

We test using a temporary CSV written from our fixture,
so no real data file is needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_temp_csv(df: pd.DataFrame, tmp_path) -> str:
    """Write a dataframe to a temp CSV and return the path."""
    path = os.path.join(tmp_path, "test_solar.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# We import after helpers so pytest path resolution works cleanly
# ---------------------------------------------------------------------------

from preprocessing import prepare_data  # noqa: E402

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_returns_five_values(self, sample_solar_df, tmp_path):
        """prepare_data() should return exactly 5 values."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        result = prepare_data(path, daylight_only=False)
        assert len(result) == 5

    def test_correct_number_of_features(self, sample_solar_df, tmp_path):
        """X_train and X_test should have exactly 4 features."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        X_train, X_test, _, _, _ = prepare_data(path, daylight_only=False)
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

    def test_train_test_split_ratio(self, sample_solar_df, tmp_path):
        """Test set should be ~20% of total data."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        X_train, X_test, _, _, _ = prepare_data(path, daylight_only=False)
        total = X_train.shape[0] + X_test.shape[0]
        test_ratio = X_test.shape[0] / total
        assert 0.15 <= test_ratio <= 0.25  # allow small rounding variance

    def test_daylight_filter_reduces_rows(self, sample_solar_df, tmp_path):
        """Daylight filtering should return fewer rows than no filter."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        X_all, _, _, _, _ = prepare_data(path, daylight_only=False)
        X_day, _, _, _, _ = prepare_data(path, daylight_only=True)
        assert X_day.shape[0] < X_all.shape[0]

    def test_scaler_is_fitted(self, sample_solar_df, tmp_path):
        """Scaler should be fitted — mean_ attribute should exist."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        _, _, _, _, scaler = prepare_data(path, daylight_only=False)
        assert hasattr(scaler, "mean_"), "Scaler has not been fitted"

    def test_scaled_data_is_normalised(self, sample_solar_df, tmp_path):
        """X_train values should be roughly zero-centred after scaling."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        X_train, _, _, _, _ = prepare_data(path, daylight_only=False)
        # Mean of scaled training data should be close to 0
        assert abs(X_train.mean()) < 0.5

    def test_no_nulls_in_output(self, sample_solar_df, tmp_path):
        """There should be no NaN values in X_train or X_test."""
        path = write_temp_csv(sample_solar_df, tmp_path)
        X_train, X_test, _, _, _ = prepare_data(path, daylight_only=False)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
