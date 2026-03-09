import os

import pandas as pd


def validate_perth_data(file_path: str | None = None) -> bool:
    """
    Run a health check on the raw Perth solar data.

    Returns True if all checks pass, False otherwise.
    """
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "raw", "perth_solar_raw.csv")

    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])

    print("--- 🔍 Perth Solar Data Health Check ---")
    all_passed = True

    # 1. Check for missing values
    null_counts = df.isnull().sum()
    print(f"\nMissing Values:\n{null_counts}")

    # 2. Daylight logic — irradiance shouldn't be high at night
    night_sun = df[
        (df["time"].dt.hour.isin([0, 1, 2, 22, 23]))
        & (df["direct_normal_irradiance"] > 0)
    ]

    if not night_sun.empty:
        print(f"\n⚠️  Alert: Found {len(night_sun)} rows with sunlight at midnight!")
        all_passed = False
    else:
        print("\n✅ Timezone alignment looks correct (no sun at night).")

    # 3. Summary stats
    print(f"\n📊 Quick Stats:\n{df.describe()}")

    return all_passed


if __name__ == "__main__":
    validate_perth_data()
