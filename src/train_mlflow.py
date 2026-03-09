import os

import joblib
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocessing import prepare_data

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "raw", "perth_solar_raw.csv"
)


def run_experiment(experiment_name: str, daylight_flag: bool) -> None:
    """Train a RandomForest and log everything to MLflow."""
    with mlflow.start_run(run_name=experiment_name):
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            DATA_PATH, daylight_only=daylight_flag
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("daylight_only_filter", daylight_flag)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        os.makedirs("artifacts", exist_ok=True)
        scaler_path = f"artifacts/scaler_{experiment_name}.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"🏁 Run '{experiment_name}' complete — MAE: {mae:.2f}, R²: {r2:.4f}")


if __name__ == "__main__":
    run_experiment("Raw_Baseline", daylight_flag=False)
    run_experiment("Daylight_Optimized", daylight_flag=True)
