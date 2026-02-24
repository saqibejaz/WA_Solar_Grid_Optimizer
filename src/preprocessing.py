import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(file_path, daylight_only=True):
    """
    Cleans, filters, and scales Perth solar data.
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Feature Engineering
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    
    # Logical Filtering based on our EDA
    if daylight_only:
        # Using the 8am-5pm window we validated
        df = df[(df['hour'] >= 8) & (df['hour'] <= 17)]
    
    features = ['temperature_2m', 'cloud_cover', 'hour', 'month']
    target = 'direct_normal_irradiance'
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler