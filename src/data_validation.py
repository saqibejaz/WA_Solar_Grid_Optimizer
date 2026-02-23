import pandas as pd

def validate_perth_data():
    file_path = r"D:\Saqib\SkillSetExpand\WA_Solar_Grid_Optimizer\data\raw/perth_solar_raw.csv"
    df = pd.read_csv(file_path)

    print("--- 🔍 Perth Solar Data Health Check ---")
    
    # 1. Check for missing values
    null_counts = df.isnull().sum()
    print(f"\nMissing Values:\n{null_counts}")

    # 2. Check for "Daylight" logic (Irradiance shouldn't be high at 2 AM!)
    # Convert time to hour
    df['time'] = pd.to_datetime(df['time'])
    night_sun = df[(df['time'].dt.hour.isin([0, 1, 2, 22, 23])) & (df['direct_normal_irradiance'] > 0)]
    
    if not night_sun.empty:
        print(f"\n⚠️ Alert: Found {len(night_sun)} rows with sunlight at midnight! Check timezone alignment.")
    else:
        print("\n✅ Timezone alignment looks correct (No sun at night).")

    # 3. Summary Stats for Outliers
    print(f"\n📊 Quick Stats:\n{df.describe()}")

if __name__ == "__main__":
    validate_perth_data()