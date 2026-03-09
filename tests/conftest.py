"""
Shared pytest fixtures available to all test files.
Fixtures are reusable pieces of test data or setup logic.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_solar_df():
    """
    A small, clean, realistic solar dataframe for testing.
    Covers a single day with hourly data (midnight to 11pm).
    """
    # hours = list(range(24))
    irradiance = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # 00:00 - 07:00 (night, no sun)
        100,
        300,
        500,
        700,
        800,  # 08:00 - 12:00 (morning ramp up)
        750,
        600,
        400,
        200,
        50,  # 13:00 - 17:00 (afternoon ramp down)
        0,
        0,
        0,
        0,
        0,
        0,  # 18:00 - 23:00 (night, no sun)
    ]
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-06-01", periods=24, freq="h"),
            "direct_normal_irradiance": irradiance,
            "cloud_cover": [20] * 24,
            "temperature_2m": [15 + i * 0.5 for i in range(24)],
        }
    )


@pytest.fixture
def dirty_solar_df():
    """
    A dataframe with deliberate problems for testing validation logic.
    Has sunlight at midnight — should trigger the night-sun alert.
    """
    # hours = list(range(24))
    irradiance = [
        500,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # 00:00 has irradiance (bad!)
        100,
        300,
        500,
        700,
        800,
        750,
        600,
        400,
        200,
        50,
        0,
        0,
        0,
        0,
        0,
        999,  # 23:00 also has irradiance (bad!)
    ]
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-06-01", periods=24, freq="h"),
            "direct_normal_irradiance": irradiance,
            "cloud_cover": [20] * 24,
            "temperature_2m": [15.0] * 24,
        }
    )
